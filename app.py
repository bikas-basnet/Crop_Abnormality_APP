from flask import Flask, request, render_template, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField, FileField
from wtforms.validators import DataRequired
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import time
import filetype
from io import BytesIO
import numpy as np  # ← FIX 1: This line was missing (required for PyTorch 2.6+)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'My passkey is 123456'  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'Uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define SEBlock for HybridShuffleNetSqueezeNet
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define HybridShuffleNetSqueezeNet
class HybridShuffleNetSqueezeNet(nn.Module):
    def __init__(self, num_classes, deep_classifier=True):
        super(HybridShuffleNetSqueezeNet, self).__init__()
        
        shufflenet = models.shufflenet_v2_x1_0(weights=None)
        squeezenet = models.squeezenet1_1(weights=None)

        self.shufflenet_features = nn.Sequential(
            shufflenet.conv1,
            shufflenet.maxpool,
            shufflenet.stage2,
            shufflenet.stage3
        )
        self.squeezenet_features = nn.Sequential(*list(squeezenet.features)[8:])

        self.transition = nn.Sequential(
            nn.Conv2d(232, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((13, 13))
        )
        self.residual_adapter = nn.Conv2d(232, 256, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if deep_classifier:
            # Used in disease-specific hybrid models (Maize, Chilli, etc.)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # Used in crop classifier (simple direct head)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        shufflenet_out = self.shufflenet_features(x)
        residual = self.residual_adapter(shufflenet_out)
        
        x = self.transition(shufflenet_out)
        residual = nn.functional.interpolate(residual, size=(13, 13), mode='bilinear', align_corners=False)
        x = x + residual
        
        x = self.squeezenet_features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# Absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# THIS ORDER MUST MATCH YOUR CROP CLASSIFIER'S TRAINING FOLDER ORDER
crop_class_names = [
    'Acid-Lime', 'Cauliflower', 'Chilli', 'Kidneybean', 'LargeCardamom',
    'Maize', 'Mungbean', 'Onion', 'Rice', 'Sesame', 'Strawberry'
]

# Fully synchronized crop configurations
crop_configs = {
    'Acid-Lime': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Mobilenet_v2_Citrus.pth'),
        'model_type': 'mobilenet_v2',
        'num_classes': 2,
        'class_names': ['Canker', 'Healthy'],
        'recommendations': {
            'Healthy': [
                {
                    'action': 'Maintain fertilization and irrigation',
                    'priority': 'Medium',
                    'details': 'Apply NPK (100:50:50 kg/ha) or farmyard manure (10 t/ha). Irrigate 1 inch/week. Maintain soil pH 6.0-6.8.',
                    'follow_up': 'Test soil annually.'
                },
                {
                    'action': 'Pest monitoring',
                    'priority': 'Low',
                    'details': 'Check for citrus leaf miners and psylla. Apply neem oil (2 ml/L water, 300 ml/ha).',
                    'follow_up': 'Inspect weekly during new leaf flush.'
                }
            ],
            'Canker': [
                {
                    'action': 'Apply copper-based bactericides',
                    'priority': 'High',
                    'details': 'Use copper oxychloride or Bordeaux mixture (3 g/L water, 2–3 kg/ha). Spray every 7–10 days during rainy season.',
                    'follow_up': 'Monitor raised corky lesions on leaves and fruit.'
                },
                {
                    'action': 'Cultural sanitation',
                    'priority': 'High',
                    'details': 'Prune and burn infected twigs. Avoid injury during harvesting. Use resistant rootstocks (e.g., Rangpur lime).',
                    'follow_up': 'Inspect regularly during monsoon.'
                }
            ]
        }
    },

    'Cauliflower': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Mobilenetv2_cauliflower.pth'),
        'model_type': 'mobilenet_v2',
        'num_classes': 4,
        'class_names': ['Alternaria', 'Healthy', 'Phosphorous Deficiency', 'Phytotoxicity'],
        'recommendations': {
            'Alternaria': [
                {'action': 'Cultural controls', 'priority': 'High', 'details': 'Use crop rotation (2–3 years with non-brassicas), remove infected debris, ensure 45x45 cm spacing.', 'follow_up': 'Monitor during humid seasons (Oct–Mar).'},
                {'action': 'Fungicide application', 'priority': 'High', 'details': 'Apply Trichoderma harzianum (5 g/L, 500 g/ha) or copper oxychloride (2.5 g/L) or mancozeb (2 g/L) every 7–10 days.', 'follow_up': 'Rotate fungicide groups to avoid resistance.'}
            ],
            'Healthy': [
                {'action': 'Balanced fertilization', 'priority': 'Medium', 'details': 'Apply NPK (120:60:80 kg/ha), farmyard manure (20 t/ha). Maintain pH 6.0–7.0. Irrigate 1–1.5 inches/week.', 'follow_up': 'Soil test every year.'},
                {'action': 'Pest prevention', 'priority': 'Medium', 'details': 'Use neem oil (2 ml/L) or Bt (1 g/L) against diamondback moth and cabbage worm.', 'follow_up': 'Weekly field scouting.'}
            ],
            'Phosphorous Deficiency': [
                {'action': 'Apply phosphorus', 'priority': 'High', 'details': 'Single superphosphate (250 kg/ha) or DAP (100 kg/ha) at planting. Side-dress 50 kg/ha P₂O₅ at 30 DAT.', 'follow_up': 'Retest soil after 4–6 weeks.'},
                {'action': 'Soil health improvement', 'priority': 'Medium', 'details': 'Add vermicompost (5 t/ha) and correct pH with lime if below 6.0.', 'follow_up': 'Watch for purple tint on leaves.'}
            ],
            'Phytotoxicity': [
                {'action': 'Stop chemical use & flush', 'priority': 'High', 'details': 'Stop herbicide/pesticide. Heavy irrigation (2–3 inches over 2 days) to leach chemicals.', 'follow_up': 'Observe recovery in 7–14 days.'},
                {'action': 'Switch to safer inputs', 'priority': 'High', 'details': 'Use only neem-based or biological pesticides. Test soil for heavy metals if issue persists.', 'follow_up': 'Monitor new growth.'}
            ]
        }
    },

    'Chilli': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Hybrid_chilli.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 3,
        'class_names': ['Anthracnose', 'Dieback', 'Healthy'],
        'recommendations': {
            'Healthy': [
                {'action': 'Nutrition & water management', 'priority': 'Medium', 'details': 'Apply NPK (100:50:50 kg/ha), FYM (10 t/ha). Irrigate 1 inch/week. pH 6.0–6.8.', 'follow_up': 'Soil test every 6 months.'},
                {'action': 'Pest monitoring', 'priority': 'Low', 'details': 'Regular check for thrips, mites, fruit borers. Use neem oil (2 ml/L) or Beauveria bassiana.', 'follow_up': 'Weekly during flowering & fruiting.'}
            ],
            'Anthracnose': [
                {'action': 'Fungicide spray', 'priority': 'High', 'details': 'Carbendazim (1 g/L) or difenoconazole (0.5 ml/L) or Trichoderma viride (5 g/L). Spray every 7–10 days from flowering.', 'follow_up': 'Look for sunken spots on fruit.'},
                {'action': 'Cultural practices', 'priority': 'High', 'details': 'Remove infected fruits, improve airflow (45x45 cm spacing), avoid overhead irrigation.', 'follow_up': 'Sanitation during monsoon.'}
            ],
            'Dieback': [
                {'action': 'Copper fungicide', 'priority': 'High', 'details': 'Copper oxychloride (3 g/L) or Bordeaux mixture every 7–10 days.', 'follow_up': 'Monitor twig drying from tip.'},
                {'action': 'Reduce plant stress', 'priority': 'High', 'details': 'Prune affected parts, avoid waterlogging, apply mulch.', 'follow_up': 'Check root health.'}
            ]
        }
    },

    'Kidneybean': {
        'model_path': os.path.join(BASE_DIR, 'model', 'squeezenet_Rajma_ND.pth'),
        'model_type': 'squeezenet1_1',
        'num_classes': 2,
        'class_names': ['Healthy', 'Nitrogen Deficiency'],
        'recommendations': {
            'Healthy': [
                {'action': 'Pest & disease monitoring', 'priority': 'Low', 'details': 'Check aphids, pod borers. Use neem oil (2 ml/L) or Bt (1 g/L).', 'follow_up': 'Weekly scouting.'},
                {'action': 'Balanced nutrition', 'priority': 'Medium', 'details': 'NPK (25:50:25 kg/ha), FYM (10 t/ha). pH 6.0–6.5.', 'follow_up': 'Soil test every season.'}
            ],
            'Nitrogen Deficiency': [
                {'action': 'Apply nitrogen', 'priority': 'High', 'details': 'Urea (50 kg/ha) basal or top-dress 25 kg/ha at flowering. Or use Rhizobium inoculation at sowing.', 'follow_up': 'Check leaf color after 10–14 days.'},
                {'action': 'Biological enhancement', 'priority': 'Medium', 'details': 'Seed treatment with Rhizobium leguminosarum (5 g/kg seed). Add vermicompost (5 t/ha).', 'follow_up': 'Check root nodules at 30 DAS.'}
            ]
        }
    },

    'LargeCardamom': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Squeez_suffle_cardamom_Hybrid.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 2,
        'class_names': ['Blight', 'Healthy'],
        'recommendations': {
            'Healthy': [
                {'action': 'Shade & nutrition', 'priority': 'Medium', 'details': 'Maintain 50–60% shade. Apply NPK (60:40:80 kg/ha) + FYM (10 t/ha). pH 4.5–6.0.', 'follow_up': 'Check shade trees annually.'},
                {'action': 'Pest monitoring', 'priority': 'Low', 'details': 'Watch for stem borers, aphids. Use neem or Beauveria bassiana.', 'follow_up': 'Weekly during monsoon.'}
            ],
            'Blight': [
                {'action': 'Fungicide application', 'priority': 'High', 'details': 'Mancozeb (2.5 g/L) or Trichoderma harzianum (5 g/L) every 7–10 days during wet weather.', 'follow_up': 'Monitor leaf spots & stem rot.'},
                {'action': 'Sanitation & spacing', 'priority': 'High', 'details': 'Remove infected plants, maintain 1.5x1.5 m spacing, improve drainage.', 'follow_up': 'Regular field inspection.'}
            ]
        }
    },

    'Maize': {
        'model_path': os.path.join(BASE_DIR, 'model', 'HybridMaizemodel.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 5,
        'class_names': ['Downy Mildew', 'Healthy', 'NCLB', 'Rust', 'SCLB'],
        'recommendations': {
            'Healthy': [
                {'action': 'Fertilization & pest control', 'priority': 'Medium', 'details': 'NPK (120:60:40 kg/ha), compost (10 t/ha). Use Bt against stem borers.', 'follow_up': 'Soil test yearly.'},
                {'action': 'Irrigation management', 'priority': 'Medium', 'details': '1–1.5 inches/week. Good drainage. pH 5.5–7.0.', 'follow_up': 'Monitor during tasseling.'}
            ],
            'Downy Mildew': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Metalaxyl + mancozeb or Trichoderma viride (5 g/L) at 15–20 DAS.', 'follow_up': 'Check for white downy growth.'},
                {'action': 'Cultural control', 'priority': 'High', 'details': 'Resistant hybrids, remove debris, 60x20 cm spacing.', 'follow_up': 'Field sanitation.'}
            ],
            'NCLB': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Chlorothalonil or mancozeb every 7–10 days.', 'follow_up': 'Monitor cigar-shaped lesions.'},
                {'action': 'Crop rotation', 'priority': 'Medium', 'details': 'Rotate with legumes, use resistant varieties.', 'follow_up': 'Annual planning.'}
            ],
            'Rust': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Azoxystrobin or propiconazole at first sign.', 'follow_up': 'Check orange pustules.'},
                {'action': 'Cultural', 'priority': 'Medium', 'details': 'Avoid dense planting, remove volunteer maize.', 'follow_up': 'During grain filling.'}
            ],
            'SCLB': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Propiconazole or mancozeb sprays.', 'follow_up': 'Monitor tan lesions with dark margins.'},
                {'action': 'Sanitation', 'priority': 'High', 'details': 'Destroy residues, crop rotation.', 'follow_up': 'Post-harvest cleanup.'}
            ]
        }
    },

    'Mungbean': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Hybridmungbean.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 2,
        'class_names': ['Healthy', 'Leaf Crinkle Viruses'],
        'recommendations': {
            'Healthy': [
                {'action': 'Irrigation & nutrition', 'priority': 'Medium', 'details': 'Light irrigation, NPK (20:40:20 kg/ha), pH 6.0–6.5.', 'follow_up': 'Avoid waterlogging.'},
                {'action': 'Pest watch', 'priority': 'Low', 'details': 'Aphids, pod borers → neem oil or Beauveria.', 'follow_up': 'Weekly during flowering.'}
            ],
            'Leaf Crinkle Viruses': [
                {'action': 'Remove infected plants', 'priority': 'High', 'details': 'Rogue out crinkled plants immediately. Use resistant varieties (Pusa Vishal, ML-267).', 'follow_up': 'Biweekly scouting.'},
                {'action': 'Vector control', 'priority': 'High', 'details': 'Whitefly control with imidacloprid (0.3 ml/L) or yellow sticky traps.', 'follow_up': 'Monitor virus spread.'}
            ]
        }
    },

    'Onion': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Hybrid_Onion_model.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 2,
        'class_names': ['Healthy', 'Purple Blotch'],
        'recommendations': {
            'Healthy': [
                {'action': 'Fertilization & irrigation', 'priority': 'Medium', 'details': 'NPK (100:50:50 kg/ha), FYM (10 t/ha). Irrigate weekly.', 'follow_up': 'Soil test yearly.'},
                {'action': 'Thrips & maggot control', 'priority': 'Low', 'details': 'Neem oil or spinosad-based insecticides.', 'follow_up': 'Weekly during bulbing.'}
            ],
            'Purple Blotch': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Mancozeb or chlorothalonil (2 g/L) every 7–10 days.', 'follow_up': 'Check purple lesions.'},
                {'action': 'Cultural', 'priority': 'High', 'details': 'Wide spacing (15x10 cm), remove debris, resistant varieties (Agrifound Dark Red).', 'follow_up': 'During humid weather.'}
            ]
        }
    },

    'Rice': {
        'model_path': os.path.join(BASE_DIR, 'model', 'MobilentV2_Rice_model.pth'),
        'model_type': 'mobilenet_v2',
        'num_classes': 5,
        'class_names': ['Blast', 'Brown Spot', 'False Smut', 'Healthy', 'Leaf Blight'],
        'recommendations': {
            'Healthy': [
                {'action': 'Water & nutrient management', 'priority': 'Medium', 'details': 'Maintain 2–5 cm water. NPK (120:40:40 kg/ha). Avoid excess FYM (Blast risk).', 'follow_up': 'Regular field leveling.'},
                {'action': 'Pest monitoring', 'priority': 'Low', 'details': 'Stem borer, leaf folder → neem or Bt.', 'follow_up': 'Weekly during tillering.'}
            ],
            'Blast': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Tricyclazole 75% WP (0.6 g/L) or Tricyclazole 22% + Hexaconazole 3% SC (1 ml/L) at tillering & panicle initiation.', 'follow_up': 'Monitor neck blast.'},
                {'action': 'Resistant varieties', 'priority': 'High', 'details': 'Terai: Sabitri, Hardinath-1, Radha-12, Sukkha series. Mid-hills: Khumal & Chandannath series.', 'follow_up': 'Avoid excess N.'}
            ],
            'Brown Spot': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Propiconazole or mancozeb sprays.', 'follow_up': 'Check oval brown spots.'},
                {'action': 'Correct deficiencies', 'priority': 'Medium', 'details': 'Balanced NPK, especially potassium.', 'follow_up': 'Soil test.'}
            ],
            'False Smut': [
                {'action': 'Fungicide at booting', 'priority': 'High', 'details': 'Copper hydroxide or propiconazole at booting stage.', 'follow_up': 'Check green-orange smut balls.'},
                {'action': 'Sanitation', 'priority': 'High', 'details': 'Remove infected panicles, avoid late planting.', 'follow_up': 'Post-harvest cleanup.'}
            ],
            'Leaf Blight': [
                {'action': 'Bactericide', 'priority': 'High', 'details': 'Streptomycin + copper or Pseudomonas fluorescens.', 'follow_up': 'Monitor water-soaked lesions.'},
                {'action': 'Cultural', 'priority': 'High', 'details': 'Resistant varieties (Masuli), proper spacing, weed control.', 'follow_up': 'During tillering.'}
            ]
        }
    },

    'Sesame': {
        'model_path': os.path.join(BASE_DIR, 'model', 'hybrd_sesame.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 2,
        'class_names': ['Healthy', 'Leaf Spot'],
        'recommendations': {
            'Healthy': [
                {'action': 'Fertilization', 'priority': 'Medium', 'details': 'NPK (40:20:20 kg/ha), light irrigation.', 'follow_up': 'Soil test every season.'},
                {'action': 'Pest control', 'priority': 'Low', 'details': 'Sesame leaf roller/webber → neem or spinosad.', 'follow_up': 'Weekly check.'}
            ],
            'Leaf Spot': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Mancozeb (2.5 g/L) or carbendazim every 7–10 days.', 'follow_up': 'Monitor circular spots.'},
                {'action': 'Cultural', 'priority': 'High', 'details': 'Remove infected leaves, wide spacing (45x15 cm), resistant varieties.', 'follow_up': 'During rainy season.'}
            ]
        }
    },

    'Strawberry': {
        'model_path': os.path.join(BASE_DIR, 'model', 'ResNet18_Weights_Strawberry.pth'),
        'model_type': 'resnet18',
        'num_classes': 2,
        'class_names': ['Alternaria Leaf Spot', 'Healthy'],
        'recommendations': {
            'Healthy': [
                {'action': 'Nutrition & irrigation', 'priority': 'Medium', 'details': 'NPK (80:60:80 kg/ha), drip irrigation, mulch, pH 5.5–6.5.', 'follow_up': 'Soil test every 6 months.'},
                {'action': 'Pest prevention', 'priority': 'Low', 'details': 'Spider mites, aphids → neem or miticide.', 'follow_up': 'Weekly during fruiting.'}
            ],
            'Alternaria Leaf Spot': [
                {'action': 'Fungicide', 'priority': 'High', 'details': 'Captan or chlorothalonil (2 g/L) every 7–10 days.', 'follow_up': 'Check purple-brown spots.'},
                {'action': 'Cultural', 'priority': 'High', 'details': 'Remove infected leaves, mulch to reduce splash, resistant varieties (Sweet Charlie).', 'follow_up': 'During wet weather.'}
            ]
        }
    }
}

# Optional: Quick access by index
def get_crop_config_by_index(idx):
    crop_name = crop_class_names[idx]
    return crop_configs[crop_name]


# Mapping to handle naming differences
crop_name_mapping = {
    'Kidneybean': 'Rajma/Kidney Bean',
    'LargeCardamom': 'Large Cardamom'
}
reverse_crop_name_mapping = {v: k for k, v in crop_name_mapping.items()}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model loading for disease models
models_cache = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(crop):
    if crop not in models_cache:
        config = crop_configs[crop]
        model_path = config['model_path']
        model_type = config['model_type']
        num_classes = config['num_classes']
        print(f"Loading model for {crop} from {model_path}")
        try:
            if model_type == 'mobilenet_v2':
                model = models.mobilenet_v2(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_type == 'squeezenet1_1':
                model = models.squeezenet1_1(weights=None)
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif model_type == 'hybrid_shufflenet_squeezenet':
                model = HybridShuffleNetSqueezeNet(num_classes=num_classes, deep_classifier=True)
            elif model_type == 'resnet18':
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models_cache[crop] = model
        except FileNotFoundError:
            print(f"Model file '{model_path}' not found.")
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    return models_cache[crop]

# Load the crop classifier (cached globally)
# GLOBAL VARIABLES — FIXED
crop_classifier = None
crop_classifier_path = os.path.join(BASE_DIR, 'model', 'CropclassifierHybrid_CLEAN.pth')

def load_crop_classifier():
    global crop_classifier
    if crop_classifier is None:
        print("Loading your final model...")
        import numpy as np
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        
        model = HybridShuffleNetSqueezeNet(num_classes=11, deep_classifier=False)  # ← False here
        checkpoint = torch.load(crop_classifier_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        crop_classifier = model
        print("CROP CLASSIFIER LOADED SUCCESSFULLY!")
    return crop_classifier

# Flask-WTF Form
class PredictForm(FlaskForm):
    crop = SelectField('Crop', choices=[('', 'Select a crop')] + [(c, c) for c in crop_configs.keys()], validators=[DataRequired()])
    file = FileField('Image', validators=[DataRequired()])

# Routes
@app.route('/')
def index():
    form = PredictForm()
    print("Rendering index with crops:", list(crop_configs.keys()))
    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    form = PredictForm()
    print("Received predict request:", request.form, request.files)
    if not form.validate_on_submit():
        print("Form validation failed:", form.errors)
        return render_template('index.html', form=form, error='Invalid form submission. Please check your inputs.')

    selected_crop = form.crop.data
    print("Selected crop:", selected_crop)
    if selected_crop not in crop_configs:
        print(f"Invalid crop selected: {selected_crop}")
        return render_template('index.html', form=form, error='Invalid crop selected.')

    config = crop_configs[selected_crop]
    class_names = config['class_names']
    recommendations = config['recommendations']

    # Handle uploaded file
    max_file_size = 10 * 1024 * 1024  # 10MB
    if form.file.data:
        file = form.file.data
        print("Uploaded file:", file.filename)
        if file.content_length > max_file_size:
            print("File size exceeds 10MB")
            return render_template('index.html', form=form, error='File size exceeds 10MB.')
        image_data = file.read()
        kind = filetype.guess(image_data)
        if kind is None or kind.mime not in ['image/jpeg', 'image/png', 'image/bmp']:
            print(f"Invalid file type: {kind.mime if kind else 'Unknown'}")
            return render_template('index.html', form=form, error='Invalid file type. Please upload an image (jpg, jpeg, png, bmp).')
        image_filename = f"uploaded_{int(time.time())}_{file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
        except Exception as e:
            print(f"Invalid image file: {str(e)}")
            return render_template('index.html', form=form, error=f'Invalid image file: {str(e)}')
    else:
        print("No image uploaded")
        return render_template('index.html', form=form, error='Please upload an image.')

    # Preprocess image
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return render_template('index.html', form=form, error=f'Error processing image: {str(e)}')

    # Run crop classifier to verify the image matches the selected crop
    try:
        classifier = load_crop_classifier()  # ← FIX 3: Now guaranteed to work
        with torch.no_grad():
            outputs = classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            crop_confidence, predicted_crop_idx = torch.max(probabilities, 1)
            predicted_crop = crop_class_names[predicted_crop_idx.item()]
            crop_confidence = crop_confidence.item() * 100

        print(f"Crop classifier prediction: {predicted_crop}, Confidence: {crop_confidence}%")

        # Handle 'Model' class (potential error in training data)
        if predicted_crop == 'Model':
            probabilities[0, crop_class_names.index('Model')] = -float('inf')
            crop_confidence, predicted_crop_idx = torch.max(probabilities, 1)
            predicted_crop = crop_class_names[predicted_crop_idx.item()]
            crop_confidence = crop_confidence.item() * 100
            print(f"Re-evaluated after ignoring 'Model': {predicted_crop}, Confidence: {crop_confidence}%")

        # Map predicted crop to config name
        predicted_crop_mapped = crop_name_mapping.get(predicted_crop, predicted_crop)
        # Map selected crop to classifier's class name
        selected_crop_classifier = reverse_crop_name_mapping.get(selected_crop, selected_crop)

        CROP_CONFIDENCE_THRESHOLD = 60.0
        if crop_confidence < CROP_CONFIDENCE_THRESHOLD:
            return render_template(
                'index.html',
                form=form,
                error=f'Image crop detection unclear (confidence {crop_confidence:.2f}%). Please upload a clearer image of {selected_crop}.',
                image_file=image_filename
            )
        if predicted_crop != selected_crop_classifier:
            return render_template(
                'index.html',
                form=form,
                error=f'Uploaded image appears to be {predicted_crop_mapped} (confidence {crop_confidence:.2f}%), not {selected_crop}. Please upload the correct crop image to avoid false results.',
                image_file=image_filename
            )
    except Exception as e:
        print(f"Error during crop classification: {str(e)}")
        return render_template('index.html', form=form, error=f'Error during crop verification: {str(e)}')

    # Proceed with disease prediction
    try:
        model = load_model(selected_crop)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = class_names[predicted.item()]
            confidence = confidence.item() * 100
        print(f"Prediction: {predicted_class}, Confidence: {confidence}%")

        # Check for unknown image
        CONFIDENCE_THRESHOLD = 50.0
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Confidence {confidence}% below threshold {CONFIDENCE_THRESHOLD}%")
            return render_template(
                'index.html',
                form=form,
                prediction='Image does not fall under the trained model.',
                image_file=image_filename
            )
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', form=form, error=f'Error during prediction: {str(e)}')

    # Determine severity
    severity = 'Mild' if confidence < 60 else 'Moderate' if confidence < 80 else 'Severe'
    selected_recommendations = recommendations.get(predicted_class, [{'action': 'Monitor crop', 'priority': 'Low', 'details': 'No action needed.', 'follow_up': 'Check weekly.'}])

    return render_template(
        'index.html',
        form=form,
        prediction=f'Predicted for {selected_crop}: {predicted_class} ({confidence:.2f}%, {severity})',
        recommendations=selected_recommendations,
        image_file=image_filename,
        confidence=confidence
    )

if __name__ == '__main__':
    # Optional: preload crop classifier on startup so first request is fast
    try:
        load_crop_classifier()
        print("Crop classifier pre-loaded successfully!")
    except Exception as e:
        print(f"Could not preload crop classifier: {e}")
    app.run(debug=True, host='0.0.0.0', port=5000)