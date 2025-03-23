Promptchan.AI/
│
├── Android App (.AAB)
│   ├── app/
│   │   ├── src/
│   │   │   ├── main/
│   │   │   │   ├── java/com/promptchan/ai/
│   │   │   │   │   ├── MainActivity.kt                 # Main entry point for the app
│   │   │   │   │   ├── AIEngine.kt                     # Core AI processing engine
│   │   │   │   │   ├── PromptchanApplication.kt        # Application class
│   │   │   │   │   ├── models/
│   │   │   │   │   │   ├── GenerationParameters.kt     # Parameters for AI generation
│   │   │   │   │   │   ├── GenerationResult.kt         # Results from AI generation
│   │   │   │   │   │   └── Character.kt                # Character model
│   │   │   │   │   ├── ui/
│   │   │   │   │   │   ├── explore/
│   │   │   │   │   │   │   ├── ExploreFragment.kt      # Browse AI-generated content
│   │   │   │   │   │   │   └── ExploreViewModel.kt
│   │   │   │   │   │   ├── create/
│   │   │   │   │   │   │   ├── CreateFragment.kt       # Create AI images
│   │   │   │   │   │   │   └── CreateViewModel.kt
│   │   │   │   │   │   ├── chat/
│   │   │   │   │   │   │   ├── ChatFragment.kt         # AI character chat
│   │   │   │   │   │   │   └── ChatViewModel.kt
│   │   │   │   │   │   ├── video/
│   │   │   │   │   │   │   ├── VideoFragment.kt        # Main video hub
│   │   │   │   │   │   │   ├── VideoViewModel.kt
│   │   │   │   │   │   │   ├── TextToVideoFragment.kt  # Text to video generation
│   │   │   │   │   │   │   ├── ImageToVideoFragment.kt # Image to video conversion
│   │   │   │   │   │   │   └── VideoEditorFragment.kt  # Video editing with AI tools
│   │   │   │   │   │   └── profile/
│   │   │   │   │   │       ├── ProfileFragment.kt      # User profile
│   │   │   │   │   │       └── ProfileViewModel.kt
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── ImageUtils.kt               # Image processing utilities
│   │   │   │   │   │   ├── VideoUtils.kt               # Video processing utilities
│   │   │   │   │   │   └── PermissionUtils.kt          # Permission handling
│   │   │   │   │   └── data/
│   │   │   │   │       ├── repository/
│   │   │   │   │       │   ├── GenerationRepository.kt # Handles AI generation requests
│   │   │   │   │       │   ├── UserRepository.kt       # User data management
│   │   │   │   │       │   └── ContentRepository.kt    # Content storage and retrieval
│   │   │   │   │       ├── api/
│   │   │   │   │       │   ├── ApiService.kt           # API interface
│   │   │   │   │       │   └── RetrofitClient.kt       # Network client
│   │   │   │   │       └── local/
│   │   │   │   │           ├── AppDatabase.kt          # Local database
│   │   │   │   │           └── PreferenceManager.kt    # User preferences
│   │   │   │   ├── res/
│   │   │   │   │   ├── layout/
│   │   │   │   │   │   ├── activity_main.xml           # Main activity layout
│   │   │   │   │   │   ├── fragment_explore.xml        # Explore screen layout
│   │   │   │   │   │   ├── fragment_create.xml         # Create screen layout
│   │   │   │   │   │   ├── fragment_chat.xml           # Chat screen layout
│   │   │   │   │   │   ├── fragment_video.xml          # Video hub layout
│   │   │   │   │   │   ├── fragment_text_to_video.xml  # Text to video layout
│   │   │   │   │   │   ├── fragment_image_to_video.xml # Image to video layout
│   │   │   │   │   │   ├── fragment_video_editor.xml   # Video editor layout
│   │   │   │   │   │   └── fragment_profile.xml        # Profile screen layout
│   │   │   │   │   ├── drawable/                       # App icons and images
│   │   │   │   │   ├── values/                         # App strings, colors, styles
│   │   │   │   │   ├── navigation/
│   │   │   │   │   │   └── nav_graph.xml               # Navigation paths
│   │   │   │   │   └── menu/
│   │   │   │   │       ├── bottom_nav_menu.xml         # Bottom navigation menu
│   │   │   │   │       └── menu_main.xml               # Main options menu
│   │   │   │   └── assets/
│   │   │   │       ├── models/                         # TensorFlow Lite models
│   │   │   │       │   ├── promptchan_ai.tflite        # Core AI model
│   │   │   │       │   ├── text_to_video.tflite        # Text to video model
│   │   │   │       │   ├── image_to_video.tflite       # Image to video model
│   │   │   │       │   └── inpainting.tflite           # AI eraser model
│   │   │   │       └── sample_data/                    # Sample content for demo
│   │   │   └── androidTest/                            # Instrumented tests
│   │   └── build.gradle                                # App module build config
│   ├── build.gradle                                    # Project build config
│   └── settings.gradle                                 # Project settings
│
├── Backend Server
│   ├── promptchan_ai_studio.py                         # Main Python Flask application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── promptchan_ai.py                            # Core AI model definition
│   │   ├── text_to_video.py                            # Text to video model
│   │   ├── image_to_video.py                           # Image to video model
│   │   └── inpainting.py                               # AI eraser model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py                         # Image utilities
│   │   ├── video_processing.py                         # Video utilities
│   │   └── file_management.py                          # File handling
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                               # Main stylesheet
│   │   ├── js/
│   │   │   ├── main.js                                 # Main JavaScript
│   │   │   ├── text-to-video.js                        # Text to video functionality
│   │   │   ├── image-to-video.js                       # Image to video functionality
│   │   │   └── ai-eraser.js                            # AI eraser functionality
│   │   └── img/                                        # Static images
│   ├── templates/
│   │   ├── index.html                                  # Home page
│   │   ├── explore.html                                # Explore page
│   │   ├── create.html                                 # Create page
│   │   ├── chat.html                                   # Chat page
│   │   ├── video.html                                  # Video hub page
│   │   ├── text_to_video.html                          # Text to video page
│   │   ├── image_to_video.html                         # Image to video page
│   │   ├── ai_eraser.html                              # AI eraser page
│   │   └── profile.html                                # User profile page
│   ├── uploads/                                        # User uploaded content
│   │   ├── images/                                     # Uploaded images
│   │   ├── videos/                                     # Uploaded videos
│   │   └── masks/                                      # Uploaded masks for AI eraser
│   ├── outputs/                                        # Generated content
│   │   ├── images/                                     # Generated images
│   │   └── videos/                                     # Generated videos
│   ├── requirements.txt                                # Python dependencies
│   └── Dockerfile                                      # Container definition
│
├── AI Models
│   ├── text_to_video/
│   │   ├── model.py                                    # Model definition
│   │   ├── train.py                                    # Training script
│   │   ├── inference.py                                # Inference script
│   │   └── checkpoints/                                # Saved model weights
│   ├── image_to_video/
│   │   ├── model.py                                    # Model definition
│   │   ├── train.py                                    # Training script
│   │   ├── inference.py                                # Inference script
│   │   └── checkpoints/                                # Saved model weights
│   ├── inpainting/
│   │   ├── model.py                                    # Model definition
│   │   ├── train.py                                    # Training script
│   │   ├── inference.py                                # Inference script
│   │   └── checkpoints/                                # Saved model weights
│   └── character_model/
│       ├── model.py                                    # Model definition
│       ├── train.py                                    # Training script
│       ├── inference.py                                # Inference script
│       └── checkpoints/                                # Saved model weights
│
└── Documentation
    ├── API_DOCS.md                                     # API documentation
    ├── USER_GUIDE.md                                   # User guide
    ├── DEVELOPER_GUIDE.md                              # Developer guide
    └── ARCHITECTURE.md                                 # Architecture overview
    
