# ===============================
# Step 6.1 Performing Training on Train Dataset -- Default Parameters
# ===============================
from cnn_root_impl.data_prep.data_preparation import load_dataset
from cnn_root_impl.evaluation.cnn_evaluation import CNN_Evaluation
from cnn_root_impl.layer_component.conv2d import Conv2D
from cnn_root_impl.layer_component.dense import Dense
from cnn_root_impl.layer_component.dropout import Dropout
from cnn_root_impl.layer_component.flatten import Flatten
from cnn_root_impl.layer_component.maxpool2d import MaxPool2D
from cnn_root_impl.layer_component.relu import ReLU
from cnn_root_impl.layer_component.softmax import Softmax
from cnn_root_impl.loss.cross_entropy import CrossEntropyLoss
from cnn_root_impl.optimizer.sgd import SGD
from cnn_root_impl.train.cnn_container import Sequential
from cnn_root_impl.train.cnn_train_pipeline import CNN_Training

# 1. Prepare dataset
# Load dataset with level_1 (1024 training samples)
x_train, y_train, x_test, y_test = load_dataset(level='level_3')

# 2. Define and Build the model
# stack the layers forming a 'model'
model = Sequential([
    Conv2D(out_channels=16, kernel_size=(3, 3), padding=0, in_channels=1, name='Conv-1', kernel_initializer='he'),
    ReLU(),
    MaxPool2D(pool_size=2, stride=2, name='MaxPool2D-1'),

    Conv2D(out_channels=64, kernel_size=(3, 3), padding=0, name='Conv-2', kernel_initializer='he'),
    ReLU(),
    MaxPool2D(pool_size=2, stride=2, name='MaxPool2D-2'),

    Flatten(),
    Dense(out_features=128, name='Dense-1'),
    ReLU(),
    Dropout(0.5),
    Dense(out_features=10, name='Dense-2'),
    Softmax()
])

# Set input shape for report generation
input_shape = (x_train.shape[0], 28, 28, 1)
model.input_shape_for_report = input_shape

# build each layers within the 'model'
model.build(input_shape=input_shape, show_summary=True)  # Input shape must be specified

# 3. Initialize Optimizer
optimizer = SGD(lr=0.001)

# 4. Initialize Loss function (criterion)
criterion = CrossEntropyLoss()

# 5. Performing Training
train_pipeline = CNN_Training(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    x_train=x_train, y_train=y_train,
    x_test=x_test, y_test=y_test,
    val_rate=0.2,  # 20% of training data for validation
    epochs=20,
    batch_size=100,
    patience=5,  # early stop patience
    report=True
)

result = train_pipeline.train()
if isinstance(result, tuple) and len(result) == 2:
    history, report_gen = result
else:
    history = result
    report_gen = None

# 6. Performing Training
evaluation = CNN_Evaluation(report_gen)
evaluation.loss_accuracy_curve(history)

# 7. Performing Prediction on Test Dataset
test_logits, test_preds, test_labels = evaluation.predict_on_test(model, x_test, y_test)

# 8. Evaluation with Confusion Matrix
evaluation.show_confusion_matrix(test_labels, test_preds)

# 9. Evaluation with Classification Report
evaluation.show_classification_report(test_labels, test_preds)

# 10. Evaluation with ROC Curves and AUCs
evaluation.show_ROC_Curves(test_labels, y_test, test_logits)

# Generate and save report if report generator is available
if report_gen:
    report_file = report_gen.generate_html_report()
    print(f"Training report generated: {report_file}")