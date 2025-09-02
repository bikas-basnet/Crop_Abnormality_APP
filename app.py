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
    def __init__(self, num_classes):
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
            nn.Conv2d(232, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((13, 13))
        )
        self.residual_adapter = nn.Conv2d(232, 256, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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

# Crop configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
crop_configs = {
    'Cauliflower': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Mobilenetv2_cauliflower.pth'),
        'model_type': 'mobilenet_v2',
        'num_classes': 4,
        'class_names': ['Alternaria', 'Healthy', 'Phosphorous Deficiency', 'Phytotoxicity'],
        'recommendations': {
            'Alternaria': [
                {
                    'action': 'Apply cultural controls',
                    'priority': 'High',
                    'details': 'Use crop rotation (2-3 years with non-brassicas), remove infected debris, and ensure 45x45 cm spacing for air circulation.',
                    'follow_up': 'Monitor during humid seasons (Oct-Mar).'
                },
                {
                    'action': 'Use biological and chemical fungicides',
                    'priority': 'High',
                    'details': 'Apply Trichoderma harzianum (5 g/L water, 500 g/ha) or copper-based fungicide (e.g., Bordeaux mixture, 2.5 kg/ha) every 7-10 days. Mix 2 ml/L water (300 ml/ha) and spray every 7–14 days as a preventative measure',
                    'follow_up': 'Rotate fungicide classes; monitor resistance.'
                }
            ],
            'Healthy': [
                {
                    'action': 'Maintain balanced fertilization',
                    'priority': 'Medium',
                    'details': 'Apply NPK (120:60:80 kg/ha) or farmyard manure (20 t/ha). Maintain soil pH 6.0-7.0 with lime (500 kg/ha if needed). Irrigate 1-1.5 inches/week.',
                    'follow_up': 'Test soil annually.'
                },
                {
                    'action': 'Prevent pests biologically',
                    'priority': 'Medium',
                    'details': 'Use neem oil (2 ml/L water, 300 ml/ha) or Bacillus thuringiensis (1 g/L water, 500 g/ha) for pests like cabbage worms.',
                    'follow_up': 'Check weekly for pests.'
                }
            ],
            'Phosphorous Deficiency': [
                {
                    'action': 'Apply phosphorus fertilizers',
                    'priority': 'High',
                    'details': 'Use single superphosphate (250 kg/ha, 16% P₂O₅) or bone meal (100 kg/ha) before planting. Side-dress 50 kg/ha at 30 days after transplanting.',
                    'follow_up': 'Retest soil after 4-6 weeks.'
                },
                {
                    'action': 'Enhance soil health',
                    'priority': 'Medium',
                    'details': 'Add vermicompost (5 t/ha) and maintain soil pH 6.0-7.0 with lime (500 kg/ha if needed).',
                    'follow_up': 'Monitor for purple leaves.'
                }
            ],
            'Phytotoxicity': [
                {
                    'action': 'Reduce chemical exposure',
                    'priority': 'High',
                    'details': 'Stop pesticide/herbicide use; switch to neem extract (2 ml/L water, 300 ml/ha). Test soil for heavy metals via extension services.',
                    'follow_up': 'Observe recovery in 1-2 weeks.'
                },
                {
                    'action': 'Flush soil',
                    'priority': 'High',
                    'details': 'Irrigate with 2-3 inches water. Water over 2 days to leach chemicals. Ensure proper drainage.',
                    'follow_up': 'Monitor for leaf burn or wilting.'
                }
            ]
        }
    },
    'Rajma/Kidney Bean': {
        'model_path': os.path.join(BASE_DIR, 'model', 'squeezenet_Rajma_ND.pth'),
        'model_type': 'squeezenet1_1',
        'num_classes': 2,
        'class_names': ['Healthy', 'Nitrogen Deficiency'],
        'recommendations': {
            'Healthy': [
                {
                    'action': 'Monitor and manage pests',
                    'priority': 'Low',
                    'details': 'Inspect for aphids and pod borers weekly. Apply neem oil (2 ml/L water, 300 ml/ha) or Bacillus thuringiensis (1 g/L water, 500 g/ha).',
                    'follow_up': 'Check pest levels after 7 days.'
                },
                {
                    'action': 'Maintain balanced fertilization',
                    'priority': 'Medium',
                    'details': 'Apply NPK (25:50:25 kg/ha) or farmyard manure (10 t/ha). Maintain soil pH 6.0-6.5 with lime (500 kg/ha if needed). Irrigate 1 inch/week.',
                    'follow_up': 'Test soil every 6 months.'
                }
            ],
            'Nitrogen Deficiency': [
                {
                    'action': 'Apply nitrogen fertilizers',
                    'priority': 'High',
                    'details': 'Apply urea (46% N, 50 kg/ha) or ammonium sulfate (20 kg/ha) at 15-20 days after sowing. Side-dress 25 kg/ha urea at flowering.',
                    'follow_up': 'Monitor yellowing leaves after 2 weeks.'
                },
                {
                    'action': 'Enhance soil health biologically',
                    'priority': 'Medium',
                    'details': 'Inoculate seeds with Rhizobium leguminosarum (5 g/kg seed). Add vermicompost (5 t/ha).',
                    'follow_up': 'Check root nodulation after 30 days.'
                }
            ]
        }
    },
    'Large Cardamom': {
        'model_path': os.path.join(BASE_DIR, 'model', 'Squeez_suffle_cardamom_Hybrid.pth'),
        'model_type': 'hybrid_shufflenet_squeezenet',
        'num_classes': 2,
        'class_names': ['Blight', 'Healthy'],
        'recommendations': {
            'Healthy': [
                {
                    'action': 'Maintain optimal shade and nutrition',
                    'priority': 'Medium',
                    'details': 'Ensure 50-60% shade with shade trees (e.g., Alnus nepalensis). Apply NPK (60:40:80 kg/ha) or farmyard manure (10 t/ha). Maintain soil pH 4.5-6.0.',
                    'follow_up': 'Check shade and soil every 6 months.'
                },
                {
                    'action': 'Monitor pests',
                    'priority': 'Low',
                    'details': 'Inspect for stem borers and aphids. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Monitor weekly during monsoon.'
                }
            ],
            'Blight': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use Trichoderma harzianum (5 g/L water, 500 g/ha) or mancozeb (2.5 g/L water, 2 kg/ha). Spray every 7-10 days during humid conditions.',
                    'follow_up': 'Monitor leaf spots and stem lesions.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected plant parts. Maintain 1.5x1.5 m spacing. Avoid waterlogging.',
                    'follow_up': 'Inspect biweekly during wet seasons.'
                }
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
                {
                    'action': 'Maintain irrigation and nutrition',
                    'priority': 'Medium',
                    'details': 'Irrigate 1 inch/week, avoiding waterlogging. Apply NPK (20:40:20 kg/ha) or farmyard manure (5 t/ha). Maintain soil pH 6.0-6.5.',
                    'follow_up': 'Check soil moisture and nutrients every 6 months.'
                },
                {
                    'action': 'Monitor pests',
                    'priority': 'Low',
                    'details': 'Inspect for aphids and pod borers. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Monitor weekly during flowering.'
                }
            ],
            'Leaf Crinkle Viruses': [
                {
                    'action': 'Cultural and resistant varieties',
                    'priority': 'High',
                    'details': 'Remove infected plants. Use resistant varieties (e.g., Pusa Vishal). Control vectors with neem oil (2 ml/L water, 300 ml/ha).',
                    'follow_up': 'Monitor crinkled leaves biweekly.'
                },
                {
                    'action': 'Vector management',
                    'priority': 'High',
                    'details': 'Apply imidacloprid (0.3 ml/L water, 200 ml/ha) for whiteflies. Use yellow sticky traps (10/ha).',
                    'follow_up': 'Check viral spread after 7-10 days.'
                }
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
                {
                    'action': 'Balanced fertilization and pest monitoring',
                    'priority': 'Medium',
                    'details': 'Apply NPK (120:60:40 kg/ha) or compost (10 t/ha). Monitor stem borers with Bacillus thuringiensis (1 g/L water, 500 g/ha).',
                    'follow_up': 'Test soil annually; check pests weekly.'
                },
                {
                    'action': 'Maintain irrigation',
                    'priority': 'Medium',
                    'details': 'Irrigate 1-1.5 inches/week with good drainage. Maintain soil pH 5.5-7.0 with lime (500 kg/ha if needed).',
                    'follow_up': 'Monitor soil moisture during tasseling.'
                }
            ],
            'Downy Mildew': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use metalaxyl (2 g/L water, 1.5 kg/ha) or Trichoderma viride (5 g/L water, 500 g/ha). Spray at 15-20 days after sowing.',
                    'follow_up': 'Monitor chlorosis after 7 days.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected debris. Use resistant varieties (e.g., Rampur Hybrid-10). Ensure 60x20 cm spacing.',
                    'follow_up': 'Check disease spread biweekly.'
                }
            ],
            'NCLB': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use chlorothalonil (2 g/L water, 1.5 kg/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Check lesions weekly.'
                },
                {
                    'action': 'Crop management',
                    'priority': 'Medium',
                    'details': 'Rotate with legumes. Remove debris. Use resistant varieties (e.g., Arun-2).',
                    'follow_up': 'Monitor during vegetative stage.'
                }
            ],
            'Rust': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use azoxystrobin (1 ml/L water, 500 ml/ha) or Trichoderma viride (5 g/L water, 500 g/ha). Spray at first pustules.',
                    'follow_up': 'Monitor spread every 7 days.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'Medium',
                    'details': 'Avoid dense planting (60x20 cm spacing). Remove volunteer plants. Use resistant hybrids.',
                    'follow_up': 'Inspect during grain filling.'
                }
            ],
            'SCLB': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use propiconazole (1 ml/L water, 500 ml/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Check spots weekly.'
                },
                {
                    'action': 'Field sanitation',
                    'priority': 'High',
                    'details': 'Destroy infected residues. Rotate with non-host crops. Use resistant varieties.',
                    'follow_up': 'Monitor during rainy season.'
                }
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
                {
                    'action': 'Maintain fertilization and irrigation',
                    'priority': 'Medium',
                    'details': 'Apply NPK (80:60:80 kg/ha) or compost (10 t/ha). Irrigate 1 inch/week with drip irrigation. Maintain soil pH 5.5-6.5.',
                    'follow_up': 'Test soil nutrients every 6 months.'
                },
                {
                    'action': 'Pest and disease prevention',
                    'priority': 'Low',
                    'details': 'Monitor spider mites and aphids. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Check weekly during fruiting.'
                }
            ],
            'Alternaria Leaf Spot': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use captan (2 g/L water, 1.5 kg/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor leaf spots biweekly.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected leaves. Use mulch to reduce soil splash. Plant resistant varieties (e.g., Sweet Charlie).',
                    'follow_up': 'Inspect during wet seasons.'
                }
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
                {
                    'action': 'Maintain irrigation and nutrition',
                    'priority': 'Medium',
                    'details': 'Irrigate 1 inch/week with proper drainage. Apply NPK (100:50:50 kg/ha) or farmyard manure (10 t/ha). Maintain soil pH 6.0-6.5.',
                    'follow_up': 'Check soil moisture and nutrients every 6 months.'
                },
                {
                    'action': 'Pest monitoring',
                    'priority': 'Low',
                    'details': 'Monitor thrips and mites. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Inspect weekly during flowering.'
                }
            ],
            'Anthracnose': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use carbendazim (1 g/L water, 500 g/ha) or Trichoderma viride (5 g/L water, 500 g/ha). Spray every 7-10 days during fruiting.',
                    'follow_up': 'Monitor fruits and leaves for dark spots.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected fruits and debris. Use resistant varieties (e.g., Pusa Jwala). Ensure 45x45 cm spacing.',
                    'follow_up': 'Check during monsoon.'
                }
            ],
            'Dieback': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use copper oxychloride (3 g/L water, 2 kg/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor plant vigor and twig dieback.'
                },
                {
                    'action': 'Improve plant health',
                    'priority': 'High',
                    'details': 'Prune affected branches. Avoid waterlogging. Apply mulch to reduce stress.',
                    'follow_up': 'Inspect biweekly during wet seasons.'
                }
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
                {
                    'action': 'Maintain water and nutrition. Over FYM used has risk of Blast suceptible',
                    'priority': 'Medium',
                    'details': 'Keep fields flooded (2-5 cm depth). Apply NPK (120:40:40 kg/ha) or farmyard manure (10 t/ha). Maintain soil pH 5.5-6.5.',
                    'follow_up': 'Check water and nutrients every 6 months.'
                },
                {
                    'action': 'Pest monitoring',
                    'priority': 'Low',
                    'details': 'Monitor stem borers and leaf folders. Apply neem oil (2 ml/L water, 300 ml/ha) or Bacillus thuringiensis (1 g/L water, 500 g/ha).',
                    'follow_up': 'Inspect weekly during tillering.'
                }
            ],
            'Blast': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': (
                        "Use Tricyclazole 22% + Hexaconazole 3% SC (1 ml/L water, 500 ml/ha, 0.2% concentration) for foliar spray. "
                        "Alternatively, use Tricyclazole 75% WP (0.6 g/L water, 300 g/ha) or Trichoderma viride (5 g/L water, 500 g/ha). "
                        "Spray at tillering and panicle initiation stages, repeating 2–3 times at 7-day intervals. "
                        "Mix 500 ml of Tricyclazole 22% + Hexaconazole 3% SC in 500 liters of water per hectare."
                    ),
                    'follow_up': 'Monitor neck and leaf lesions.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': (
                        "Use resistant varieties (e.g., Sabitri). In Nepal's Terai region (60–900 masl), leaf blast-resistant rice varieties include "
                        "Sabitri, Hardinath-1, Radha-12, Saga 4, INH12023, DR-11, Hardinath-3,Sukkha-1, Sukkha-2, Sukkha-3, Sukkha-4, Sukkha-5, and Sukkha-6, with Chaite-5, INH14120, and INH14172 showing moderate resistance."
                        "In the Midhills (1,100–1,500 masl), resistant varieties include Khumal-1, Khumal-2, Khumal-3, Chandannath-1, Chandannath-3, "
                        "Palung-2, Manjushree, IR 87760-15-2-2-4, IR 70210-39-CPA-7-1, NR 11105-B-B-20-2-1, and Khumal-13. Avoid excess nitrogen. "
                        "Remove infected debris."
                    ),
                    'follow_up': 'Check during humid conditions.'
                }
            ],
            'Brown Spot': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use propiconazole (1 ml/L water, 500 ml/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor brown spots on leaves.'
                },
                {
                    'action': 'Improve nutrition',
                    'priority': 'Medium',
                    'details': 'Correct nutrient deficiencies with NPK (120:40:40 kg/ha). Use resistant varieties (e.g., Radha-4).',
                    'follow_up': 'Check during vegetative growth.'
                }
            ],
            'False Smut': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use copper hydroxide (2 g/L water, 1.5 kg/ha) or Trichoderma viride (5 g/L water, 500 g/ha). Spray at booting stage.',
                    'follow_up': 'Monitor panicles for smut balls.'
                },
                {
                    'action': 'Field sanitation',
                    'priority': 'High',
                    'details': 'Remove infected panicles. Use resistant varieties. Avoid late planting.',
                    'follow_up': 'Inspect during flowering.'
                }
            ],
            'Leaf Blight': [
                {
                    'action': 'Apply bactericides',
                    'priority': 'High',
                    'details': 'Use streptomycin (0.15 g/L water, 100 g/ha) or Pseudomonas fluorescens (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor leaf symptoms biweekly.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Use resistant varieties (e.g., Masuli). Avoid dense planting (20x15 cm spacing). Remove weeds.',
                    'follow_up': 'Check during tillering.'
                }
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
                {
                    'action': 'Maintain fertilization and irrigation',
                    'priority': 'Medium',
                    'details': 'Apply NPK (40:20:20 kg/ha) or compost (5 t/ha). Irrigate 1 inch/week. Maintain soil pH 6.0-6.5.',
                    'follow_up': 'Test soil every 6 months.'
                },
                {
                    'action': 'Pest monitoring',
                    'priority': 'Low',
                    'details': 'Check sesame bugs and leaf rollers. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Inspect weekly during pod formation.'
                }
            ],
            'Leaf Spot': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use mancozeb (2.5 g/L water, 2 kg/ha) or Trichoderma harzianum (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor leaf spots biweekly.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected leaves. Use resistant varieties (e.g., NS-970). Ensure 45x15 cm spacing.',
                    'follow_up': 'Check during monsoon.'
                }
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
                {
                    'action': 'Maintain fertilization and irrigation',
                    'priority': 'Medium',
                    'details': 'Apply NPK (100:50:50 kg/ha) or farmyard manure (10 t/ha). Irrigate 1 inch/week. Maintain soil pH 6.0-6.8.',
                    'follow_up': 'Test soil annually.'
                },
                {
                    'action': 'Pest monitoring',
                    'priority': 'Low',
                    'details': 'Check thrips and onion maggots. Apply neem oil (2 ml/L water, 300 ml/ha) or Beauveria bassiana (2 g/L water, 500 g/ha).',
                    'follow_up': 'Inspect weekly during bulb formation.'
                }
            ],
            'Purple Blotch': [
                {
                    'action': 'Apply fungicides',
                    'priority': 'High',
                    'details': 'Use chlorothalonil (2 g/L water, 1.5 kg/ha) or Trichoderma viride (5 g/L water, 500 g/ha). Spray every 7-10 days.',
                    'follow_up': 'Monitor bulbs and leaves for purple lesions.'
                },
                {
                    'action': 'Cultural controls',
                    'priority': 'High',
                    'details': 'Remove infected debris. Use resistant varieties (e.g., Agrifound Dark Red). Ensure 15x10 cm spacing.',
                    'follow_up': 'Check during wet seasons.'
                }
            ]
        }
    }
}

# Crop classifier class names based on training folder order
crop_class_names = [
    'Cauliflower', 'Chilli', 'Kidneybean', 'LargeCardamom', 'Maize',
    'Mungbean', 'Onion', 'Rice', 'Sesame', 'Strawberry'
]

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
                model = HybridShuffleNetSqueezeNet(num_classes=num_classes)
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
crop_classifier = None
crop_classifier_path = os.path.join(BASE_DIR, 'model', 'CropclassifierHybrid.pth')

def load_crop_classifier():
    global crop_classifier
    if crop_classifier is None:
        print(f"Loading crop classifier from {crop_classifier_path}")
        try:
            model = HybridShuffleNetSqueezeNet(num_classes=len(crop_class_names))  # 11 classes
            model.load_state_dict(torch.load(crop_classifier_path, map_location=device))
            model.to(device)
            model.eval()
            crop_classifier = model
        except FileNotFoundError:
            raise FileNotFoundError(f"Crop classifier model '{crop_classifier_path}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading crop classifier: {str(e)}")
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
        classifier = load_crop_classifier()
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
    app.run(debug=True, host='0.0.0.0', port=5000)