# 📚 Customer Prioritization AI System - Complete Documentation

## 🎯 **System Overview**

The Customer Prioritization AI System is a reinforcement learning-based solution that helps debt collectors optimize their daily customer visit sequences. The system learns from historical visit data to predict which customers should be prioritized for maximum collection success.

### **Key Capabilities:**
- ✅ **Intelligent Customer Ranking**: Prioritizes customers based on success probability
- ✅ **Collector-Specific Optimization**: Adapts to individual collector strengths
- ✅ **Scalable Processing**: Handles thousands of collectors and millions of records
- ✅ **Real-time Inference**: Provides instant prioritization for daily planning
- ✅ **Stable Training**: Emergency stabilization prevents training failures

---

## 📁 **File Structure & Architecture**

```
customer_prioritization_ai/
├── 📄 config.py                    # System configuration and hyperparameters
├── 🎮 environment.py               # RL training environment (game simulator)
├── 🧠 attention_model.py           # Neural network model architecture
├── 👨‍🏫 trainer.py                    # Training engine with emergency stabilization
├── 🔄 experience_replay.py         # Memory system for stable learning
├── 🛠️ utils.py                      # Data processing and evaluation utilities
├── 🚀 main.py                      # Main execution script and system orchestrator
├── 📊 Data/                        # Training data directory
│   └── Data_30-APR-25_test.csv     # Sample training data
├── 💾 model/                       # Saved models directory
│   └── trained_customer_prioritization_model.pth
└── 📂 checkpoints/                 # Training checkpoints
    ├── best_stabilized_model.pth
    └── checkpoint_episode_*.pth
```

---

# 📄 **File-by-File Documentation**

## 1. **config.py** - System Configuration Hub

### **Purpose:**
Central configuration file containing all hyperparameters, model settings, and system constants. Acts as the control center for the entire system.

### **Key Components:**

#### **Model Architecture Settings:**
```python
CUSTOMER_FEATURES = 29      # Number of customer-related features
COLLECTOR_FEATURES = 5      # Number of collector-related features  
TEMPORAL_FEATURES = 3       # Time-based features (days to PTP, etc.)
HIDDEN_DIM = 64            # Neural network hidden layer size
DROPOUT_RATE = 0.3         # Regularization to prevent overfitting
```

#### **Emergency Stabilization Parameters:**
```python
LEARNING_RATE = 1e-4           # Conservative learning rate for stability
GRADIENT_CLIP_NORM = 0.3       # Aggressive gradient clipping
STABILITY_THRESHOLD = 5.0      # Gradient explosion detection threshold
MAX_GRADIENT_EXPLOSIONS = 10   # Training halt limit
REWARD_NORMALIZATION_FACTOR = 50.0  # Scales rewards for stable gradients
```

#### **Training Configuration:**
```python
NUM_EPISODES = 1000           # Maximum training episodes
BATCH_SIZE = 16              # Small batches for stability
REPLAY_BUFFER_SIZE = 500     # Experience memory capacity
VALIDATION_FREQ = 25         # How often to validate performance
```

### **Usage in Prioritization:**
- **Model Creation**: Defines network architecture size and complexity
- **Training Stability**: Prevents gradient explosions that were breaking training
- **Performance Tuning**: Balances learning speed vs stability
- **Production Settings**: Ensures consistent behavior across environments

### **Key Methods:**
- `set_random_seeds()`: Ensures reproducible results
- `validate_config()`: Checks for potential stability issues
- `get_emergency_config()`: Provides ultra-safe fallback settings

---

## 2. **environment.py** - RL Training Environment

### **Purpose:**
Creates a simulated training environment where the AI can practice customer prioritization decisions. Acts like a "video game" where the AI learns by trying different customer ordering strategies and receiving feedback.

### **Core Class: CollectionEnvironment**

#### **Initialization:**
```python
def __init__(self, data_df, reward_config=None):
    """
    Creates training episodes from historical visit data
    Each episode = one collector's customers for practice
    """
```

#### **Key Methods:**

##### **Episode Creation:**
```python
def _create_episodes(self):
    """
    Groups data by collector to create realistic training scenarios
    Each episode contains:
    - One collector's customer list
    - Customer features (debt amount, payment history, etc.)
    - Collector features (experience, success rate, etc.)
    - Temporal features (days since last visit, etc.)
    """
```

##### **Environment Reset:**
```python
def reset(self):
    """
    Starts a new training episode
    Returns: feature_matrix [num_customers, total_features]
    
    Like dealing a new hand of cards - gives AI a fresh
    customer prioritization challenge to solve
    """
```

##### **Reward Calculation:**
```python
def calculate_reward(self, rankings):
    """
    Evaluates how good the AI's customer ranking was
    
    Reward Components:
    - Base reward for successful visits
    - Bonus for successful collections  
    - Position weighting (higher priority = more important)
    - Penalty for unsuccessful high-priority visits
    
    Returns: numerical score indicating ranking quality
    """
```

### **Usage in Prioritization:**

#### **Training Phase:**
1. **Episode Generation**: Creates thousands of realistic collector scenarios
2. **Feature Preparation**: Converts raw data into ML-ready format
3. **Reward Feedback**: Teaches AI which prioritization strategies work

#### **Key Data Flow:**
```
Raw Visit Data → Episodes → Feature Matrix → AI Decision → Reward → Learning
```

#### **Business Logic Integration:**
- **Position Weighting**: First customer visited = most important
- **Success Tracking**: Rewards actual collection outcomes
- **Collector Context**: Each episode represents one collector's real scenario

### **Example Usage:**
```python
# Create environment from historical data
environment = CollectionEnvironment(visit_data_df)

# Training loop
for episode in range(1000):
    state = environment.reset()           # Get customer features
    action = model.predict(state)         # AI chooses ranking
    reward = environment.calculate_reward(action)  # Evaluate performance
    model.learn(state, action, reward)    # Improve strategy
```

---

## 3. **attention_model.py** - Neural Network Architecture

### **Purpose:**
Defines the AI brain that learns customer prioritization. Uses a simplified feedforward neural network designed for stability and reliability rather than complexity.

### **Core Class: AttentionRanker**

#### **Architecture Design:**
```python
class AttentionRanker(nn.Module):
    """
    Stable customer ranking model (SimpleFeedforward)
    
    Key Features:
    - No attention mechanism (prevents gradient explosion)
    - Processes customers independently 
    - Stable gradients for large customer groups
    - Bounded outputs for numerical stability
    """
```

#### **Network Structure:**
```python
self.layers = nn.Sequential(
    nn.LayerNorm(total_features),          # Input normalization
    nn.Linear(total_features, hidden_dim), # First layer
    nn.ELU(),                              # Bounded activation
    nn.Dropout(dropout_rate),              # Regularization
    nn.Linear(hidden_dim, hidden_dim//2),  # Second layer
    nn.ELU(),
    nn.Dropout(0.4),
    nn.Linear(hidden_dim//2, hidden_dim//4), # Third layer
    nn.ELU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_dim//4, 1),           # Output layer
    nn.Tanh()                              # Bound output to [-1, 1]
)
```

#### **Key Methods:**

##### **Forward Pass:**
```python
def forward(self, x, return_attention=False):
    """
    Processes customer features to generate priority scores
    
    Input: [batch_size, num_customers, features]
    Output: [batch_size, num_customers] priority scores
    
    Each customer gets an independent score used for ranking
    """
```

##### **Customer Scoring:**
```python
def get_customer_score(self, customer_features):
    """
    Scores a single customer for priority ranking
    
    Returns: Priority score between -1.0 and 1.0
    - Higher scores = higher priority
    - Lower scores = lower priority
    """
```

### **Usage in Prioritization:**

#### **Training Phase:**
- **Feature Processing**: Converts 37-dimensional customer data into priority scores
- **Stable Learning**: Bounded outputs prevent training explosions
- **Batch Processing**: Handles multiple customers simultaneously

#### **Inference Phase:**
```python
# Daily customer prioritization
customer_features = prepare_daily_customers(collector_id, date)
scores = model(customer_features)  # Get priority scores
rankings = torch.argsort(scores, descending=True)  # Rank by score
priority_list = [customers[i] for i in rankings]  # Ordered visit list
```

#### **Why This Architecture:**
- **Stability First**: Prevents the gradient explosions that broke training
- **Scalability**: Handles 2-340 customers per collector efficiently  
- **Interpretability**: Clear priority scores for business understanding
- **Production Ready**: Fast inference for real-time daily planning

---

## 4. **trainer.py** - Emergency Stabilized Training Engine

### **Purpose:**
Manages the AI training process with advanced stability features. Includes real-time monitoring, emergency intervention, and adaptive learning rate adjustment to prevent training failures.

### **Core Class: RLTrainer**

#### **Emergency Stabilization Features:**
```python
class RLTrainer:
    """
    COMPLETE STABILIZED trainer with comprehensive monitoring
    
    Emergency Features:
    - Real-time gradient explosion detection
    - Automatic learning rate reduction
    - Emergency mode activation for severe instability
    - Comprehensive training metrics and visualization
    """
```

#### **Key Components:**

##### **Initialization:**
```python
def __init__(self, model, environment, learning_rate=1e-4, batch_size=16, gradient_clip_norm=0.3):
    """
    Sets up emergency-safe training configuration
    - Conservative learning rate (1e-4 vs original 5e-3)
    - Small batch sizes (16 vs original 128)
    - Aggressive gradient clipping (0.3 vs original 0.7)
    """
```

##### **Reward Normalization:**
```python
def _normalize_reward(self, reward):
    """
    Normalizes high rewards (~100) to stable range [-2, 2]
    Critical for preventing gradient explosions
    
    Uses: tanh(reward / normalization_factor)
    """
```

##### **Stabilized Loss Function:**
```python
def _calculate_policy_loss(self, scores, action, reward):
    """
    EMERGENCY STABILIZED policy gradient loss
    
    Fixes:
    - Reward normalization prevents extreme gradients
    - Multiple clipping stages prevent accumulation
    - Focus on top 3 positions reduces noise
    - Numerical stability safeguards throughout
    """
```

##### **Emergency Monitoring:**
```python
def _emergency_stability_check(self, loss_value, grad_norm):
    """
    Real-time stability monitoring and intervention
    
    Responses:
    - Gradient > 5.0: Skip update, reduce learning rate
    - Loss > 50.0: Emergency learning rate reduction
    - NaN/Inf: Skip update completely
    - Multiple explosions: Activate emergency mode
    """
```

#### **Training Loop:**
```python
def train(self, num_episodes=1000, validation_freq=25, patience=200):
    """
    Complete stabilized training with:
    - Real-time gradient monitoring
    - Emergency intervention system
    - Comprehensive progress tracking
    - Automatic early stopping
    - Model checkpointing
    """
```

### **Usage in Prioritization:**

#### **Training Process:**
1. **Data Loading**: Processes historical visit records
2. **Environment Setup**: Creates training scenarios
3. **Stabilized Learning**: Trains with emergency safeguards
4. **Validation**: Regular performance checks
5. **Model Saving**: Saves best performing model

#### **Emergency Intervention Example:**
```
🚨 GRADIENT EXPLOSION #1: 6.367
   Threshold: 5.0
🚑 EMERGENCY MODE ACTIVATED
   Learning rate: 1e-4 → 1e-5 (10x reduction)
✅ Stability restored - Exiting emergency mode
```

#### **Success Metrics Tracking:**
- **Gradient Norms**: Keep under 5.0 for stability
- **Loss Values**: Maintain 3-10 range for optimal learning
- **Reward Progression**: Monitor business performance improvement
- **Emergency Events**: Track and minimize instability incidents

---

## 5. **experience_replay.py** - Memory System

### **Purpose:**
Implements a smart memory system that stores past customer prioritization decisions and outcomes. Enables the AI to learn from diverse experiences rather than just the latest decision, significantly improving learning stability and efficiency.

### **Core Class: ExperienceReplay**

#### **Memory Buffer Design:**
```python
class ExperienceReplay:
    """
    Circular buffer storing customer prioritization experiences
    
    Each experience contains:
    - State: Customer feature matrix
    - Action: Chosen customer ranking
    - Reward: Success outcome
    - Episode info: Collector, date, metadata
    """
```

#### **Key Methods:**

##### **Experience Storage:**
```python
def push(self, state, action, reward, episode_info=None):
    """
    Stores a customer prioritization decision and outcome
    
    Example stored experience:
    {
        'state': customer_features_matrix,    # 90 customers x 37 features
        'action': [2, 0, 4, 1, 3, ...],      # Chosen visit order
        'reward': 8.5,                       # Collection success score
        'episode_info': {
            'collector_id': 'C001',
            'date': '2024-04-15',
            'num_customers': 90
        }
    }
    """
```

##### **Smart Sampling:**
```python
def sample(self, batch_size):
    """
    Random sampling breaks correlation between sequential experiences
    Returns diverse batch for stable learning
    """

def sample_prioritized(self, batch_size, high_reward_ratio=0.3):
    """
    Samples with preference for successful strategies
    Helps model learn from best practices more frequently
    """
```

##### **Performance Analytics:**
```python
def analyze_collector_performance(self):
    """
    Analyzes success patterns by collector
    Returns insights into individual collector strengths
    """
```

### **Usage in Prioritization:**

#### **During Training:**
1. **Experience Collection**: Every prioritization decision is stored
2. **Diverse Learning**: AI learns from random mix of past experiences  
3. **Success Pattern Recognition**: High-reward strategies are emphasized
4. **Stability Enhancement**: Breaks correlation between similar decisions

#### **Memory Management:**
- **Circular Buffer**: Automatically removes old experiences when full
- **Efficient Storage**: Numpy arrays for memory optimization
- **Statistics Tracking**: Monitors buffer health and diversity

#### **Learning Enhancement:**
```python
# Traditional RL (problematic):
for episode in training:
    decision = make_decision()
    outcome = get_outcome(decision)
    learn_from_single_experience(decision, outcome)  # Forgets previous lessons

# With Experience Replay (our approach):
for episode in training:
    decision = make_decision()
    outcome = get_outcome(decision)
    store_experience(decision, outcome)              # Remember for later
    
    # Learn from diverse past experiences
    batch = memory.sample(32)  # 32 different scenarios
    learn_from_batch(batch)    # Rich, stable learning
```

---

## 6. **utils.py** - Data Processing & Evaluation Utilities

### **Purpose:**
Provides essential data processing, model evaluation, and utility functions. Handles the conversion between raw business data and ML-ready formats, plus comprehensive model performance analysis.

### **Key Functions:**

#### **Data Loading & Preparation:**
```python
def load_and_prepare_data(data_input):
    """
    Converts raw visit data into ML-ready format
    
    Handles:
    - DataFrame or text string input
    - Feature vector parsing and cleaning
    - Numeric column conversion
    - Data validation and filtering
    
    Input: Raw CSV data with columns like:
    - COLLECTOR_ID, VISIT_DATE, CUS_FEATURE_VECTOR
    - COLL_FEATURE_VECTOR, VISIT_SUCCESS_FLAG, etc.
    
    Output: Clean DataFrame ready for environment.py
    """
```

#### **Feature Cleaning:**
```python
def _clean_customer_features(vector_str):
    """
    Parses and cleans customer feature vectors
    
    Handles:
    - String to numpy array conversion
    - NaN/Inf value replacement
    - Dimension validation and padding
    - Type conversion and normalization
    
    Ensures consistent 29-dimensional customer features
    """

def _clean_collector_features(vector_str):
    """
    Similar processing for collector features
    Ensures consistent 5-dimensional collector features
    """
```

#### **Model Evaluation:**
```python
def evaluate_model(model, environment, num_episodes=100):
    """
    Comprehensive model performance evaluation
    
    Tests model on unseen scenarios and returns:
    - Average reward performance
    - Reward distribution statistics
    - Consistency metrics
    - Performance range analysis
    
    Critical for validating model quality before deployment
    """
```

#### **Customer Prioritization Interface:**
```python
def get_customer_priorities(model, customer_features, return_details=False):
    """
    Main interface for real-time customer prioritization
    
    Input: Customer feature matrix
    Output: Priority rankings and scores
    
    This is the function called in production for daily planning:
    
    priorities = get_customer_priorities(model, daily_customers)
    # Returns: [1, 3, 2, 5, 4] - visit order by priority
    """
```

#### **Analysis & Reporting:**
```python
def create_feature_importance_report(model, sample_data):
    """
    Analyzes which customer/collector features drive decisions
    Provides business insights into AI decision-making
    """

def analyze_model_predictions(model, environment, num_samples=50):
    """
    Deep analysis of model behavior patterns
    Identifies decision strategies and performance drivers
    """
```

### **Usage in Prioritization:**

#### **Data Pipeline:**
```
Raw CSV → load_and_prepare_data() → Clean DataFrame → Environment → Training
```

#### **Production Usage:**
```python
# Daily customer prioritization workflow
raw_customers = get_daily_customer_list(collector_id, date)
clean_features = prepare_customer_features(raw_customers)
priorities = get_customer_priorities(model, clean_features)
visit_order = apply_priorities_to_customers(raw_customers, priorities)
```

#### **Performance Monitoring:**
```python
# Regular model evaluation
results = evaluate_model(model, test_environment)
print(f"Model performance: {results['avg_reward']:.3f}")

# Feature importance analysis
importance = create_feature_importance_report(model, sample_data)
print("Key decision factors:", importance['top_features'])
```

---

## 7. **main.py** - System Orchestrator

### **Purpose:**
Main execution script that orchestrates the entire training and evaluation process. Provides both command-line interface and programmatic API for the customer prioritization system.

### **Core Components:**

#### **Main Training Function:**
```python
def main(input_data=None):
    """
    Complete training pipeline orchestration:
    
    1. Data loading and preparation
    2. Environment setup
    3. Model creation
    4. Training execution
    5. Evaluation and analysis
    6. Model saving and deployment preparation
    """
```

#### **System Integration:**
```python
# Component initialization
Config.set_random_seeds()                    # Reproducible results
df = load_and_prepare_data(input_data)       # Data processing
environment = CollectionEnvironment(df)      # Training environment
model = AttentionRanker()                    # AI model
trainer = RLTrainer(model, environment)      # Training engine
```

#### **Training Execution:**
```python
trainer.train(
    num_episodes=Config.NUM_EPISODES,
    epsilon_start=Config.EPSILON_START,
    epsilon_end=Config.EPSILON_END,
    validation_freq=Config.VALIDATION_FREQ
)
```

#### **Evaluation & Demonstration:**
```python
# Model performance evaluation
evaluation_results = evaluate_model(model, environment)

# Live demonstration of prioritization
demonstrate_prioritization(model, environment)

# Business analysis
analysis_results = analyze_model_predictions(model, environment)
```

### **Usage Examples:**

#### **Training with CSV Data:**
```python
# Load and train with specific data file
df = pd.read_csv("Data/Data_30-APR-25_test.csv")
system, results = main(df)
```

#### **Production Deployment:**
```python
# Create production-ready system
system = CustomerPrioritizationSystem()
system.train(historical_data)

# Daily usage
priorities = system.prioritize_customers(
    customer_data=todays_customers,
    collector_id="C001"
)
```

---

# 🎯 **How Components Work Together in Prioritization**

## **Training Phase Flow:**

```
1. main.py loads historical visit data
2. utils.py cleans and prepares features
3. environment.py creates training episodes
4. attention_model.py processes customer features
5. trainer.py orchestrates learning with stability
6. experience_replay.py stores and samples experiences
7. config.py provides all necessary parameters
```

## **Production Inference Flow:**

```
1. Load trained model from main.py
2. Get daily customer list for collector
3. utils.py prepares customer features
4. attention_model.py generates priority scores
5. utils.get_customer_priorities() returns rankings
6. Business system applies visit order
```

## **Key Data Transformations:**

### **Raw Data → Training Format:**
```
CSV Record:
COLLECTOR_ID: C001
VISIT_DATE: 2024-04-15
CUS_FEATURE_VECTOR: [debt_amount, payment_history, ...]
VISIT_SUCCESS_FLAG: 1

↓ (utils.py processing)

Training Episode:
{
  'collector_id': 'C001',
  'customers': numpy.array([[features...], [features...]]),
  'success_outcomes': [1, 0, 1, 1, 0]
}
```

### **Features → Priority Scores:**
```
Customer Features (37 dimensions):
[1000.0, 0.8, 5, 2, ...]  # debt, history, days, etc.

↓ (attention_model.py)

Priority Score:
0.847  # Higher = visit first

↓ (ranking)

Visit Position:
2  # Second customer to visit today
```

## **Business Integration Points:**

### **Daily Planning Workflow:**
1. **Morning**: Get customer list for each collector
2. **AI Processing**: Generate priority scores for all customers
3. **Route Optimization**: Order customers by priority + geography
4. **Collector Assignment**: Provide prioritized visit list
5. **Evening**: Record visit outcomes for continuous learning

### **Performance Monitoring:**
1. **Weekly**: Evaluate model performance on recent data
2. **Monthly**: Retrain with new data for adaptation
3. **Quarterly**: Analyze feature importance and business insights
4. **Annually**: Full system review and architecture updates

---

# 🚀 **Deployment & Production Usage**

## **System Requirements:**
- **Python 3.7+** with PyTorch, pandas, numpy
- **Memory**: 8GB+ RAM for large datasets
- **Storage**: 1GB+ for models and checkpoints
- **CPU**: Multi-core recommended for training

## **Quick Start Guide:**

### **1. Training a New Model:**
```python
# Load your data
df = pd.read_csv("your_visit_data.csv")

# Train the system
from main import main
system, results = main(df)

print(f"Training completed! Best reward: {results['best_reward']}")
```

### **2. Daily Customer Prioritization:**
```python
# Load trained model
system = CustomerPrioritizationSystem()
system.load_model("model/trained_customer_prioritization_model.pth")

# Get today's customers for collector C001
todays_customers = get_customer_list("C001", "2024-04-16")

# Generate priorities
priorities = system.prioritize_customers(
    customer_data=todays_customers,
    collector_id="C001",
    return_details=True
)

# Use results
print("Visit order:", priorities['recommended_order'])
print("Priority scores:", priorities['scores'])
```

### **3. Performance Monitoring:**
```python
# Evaluate model performance
results = evaluate_model(model, test_environment)
print(f"Current performance: {results['avg_reward']:.3f}")

# Analyze decision patterns
analysis = analyze_model_predictions(model, environment)
print("Performance by group size:", analysis['avg_reward_by_size'])
```

---

# 📊 **Configuration & Customization**

## **Key Configuration Options:**

### **For Better Performance (if stable):**
```python
Config.LEARNING_RATE = 2e-4      # Slightly faster learning
Config.HIDDEN_DIM = 128          # Larger model capacity
Config.NUM_EPISODES = 2000       # More training time
```

### **For Better Stability (if issues):**
```python
Config.LEARNING_RATE = 5e-5      # More conservative
Config.GRADIENT_CLIP_NORM = 0.1  # Tighter gradient control
Config.BATCH_SIZE = 8            # Smaller batches
```

### **For Large Datasets:**
```python
Config.REPLAY_BUFFER_SIZE = 2000 # More experience storage
Config.VALIDATION_FREQ = 50      # Less frequent validation
Config.NUM_EPISODES = 3000       # Extended training
```

## **Emergency Troubleshooting:**

### **If Training Crashes:**
```python
Config.apply_emergency_config()  # Ultra-safe settings
# Then restart training
```

### **If Performance Degrades:**
```python
# Check model evaluation
results = evaluate_model(model, environment)
# Retrain with more data or different parameters
```

### **If Memory Issues:**
```python
Config.BATCH_SIZE = 8           # Reduce memory usage
Config.REPLAY_BUFFER_SIZE = 200 # Smaller buffer
# Process data in chunks
```

---

# 🎉 **Success Metrics & KPIs**

## **Technical Success Indicators:**
- ✅ **Training Completion**: Episodes finish without crashes
- ✅ **Stable Gradients**: Gradient norms < 5.0 
- ✅ **Learning Progress**: Rewards improve over time
- ✅ **Low Emergency Events**: < 5 gradient explosions per 1000 episodes

## **Business Success Indicators:**
- ✅ **Clear Prioritization**: Distinct priority scores for customers
- ✅ **Collector Adoption**: Collectors find recommendations useful
- ✅ **Performance Improvement**: Higher collection success rates
- ✅ **Operational Efficiency**: Reduced planning time, optimized routes

## **Monitoring Dashboard Metrics:**
```python
# Daily KPIs
model_performance = evaluate_model(model, recent_data)
print(f"Model accuracy: {model_performance['avg_reward']:.3f}")
print(f"Prediction consistency: {model_performance['std_reward']:.3f}")

# Weekly analysis
collector_analysis = analyze_collector_performance(experience_buffer)
print(f"Top performing collector strategies: {collector_analysis}")
```

This comprehensive documentation provides everything needed to understand, deploy, and maintain the Customer Prioritization AI System successfully! 🚀
