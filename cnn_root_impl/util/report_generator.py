import os
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import numpy as np


class ReportGenerator:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.report_dir = os.path.join(results_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_info": {},
            "model_summary": {},
            "optimizer_info": {},
            "criterion_info": {},
            "training_params": {},
            "training_history": {},
            "evaluation_results": {}
        }
        
    def add_data_info(self, x_train_shape, y_train_shape, x_test_shape, y_test_shape, level=None):
        self.report_data["data_info"] = {
            "train_shape": x_train_shape,
            "train_labels_shape": y_train_shape,
            "test_shape": x_test_shape,
            "test_labels_shape": y_test_shape,
            "dataset_level": level
        }
        
    def add_model_summary(self, model):
        layers_info = []
        total_params = 0
        
        # 获取模型的输入形状
        input_shape = getattr(model, 'input_shape_for_report', None)
        if input_shape is None:
            # 如果没有设置input_shape_for_report，尝试其他方式获取
            if hasattr(model, '_output_shape') and model.layers:
                # 使用一个默认的输入形状
                input_shape = 'N/A'
            else:
                input_shape = 'N/A'
        
        shape = input_shape
        
        for i, layer in enumerate(model.layers):
            input_shape_layer = shape
            
            # 计算当前层的输出形状
            try:
                if hasattr(layer, 'output_shape') and input_shape_layer != 'N/A':
                    out_shape = layer.output_shape(input_shape_layer)
                    # 更新shape供下一层使用
                    shape = out_shape
                else:
                    # 尝试从层获取预计算的输出形状
                    if hasattr(layer, '_output_shape'):
                        out_shape = layer._output_shape
                    else:
                        out_shape = 'N/A'
            except Exception as e:
                # 如果计算输出形状时出错，设为N/A
                out_shape = 'N/A'
            
            # 计算参数数量
            params_count = 0
            if hasattr(layer, 'params') and layer.params:
                for p in layer.params:
                    # 使用np.prod来正确计算参数数量，与_summary函数一致
                    params_count += np.prod(p.shape)
                    
            total_params += params_count
            
            layers_info.append({
                "index": i+1,
                "type": layer.__class__.__name__,
                "input_shape": str(input_shape_layer),
                "output_shape": str(out_shape),
                "params_count": params_count,
                "name": getattr(layer, 'name', 'N/A')
            })
            
        self.report_data["model_summary"] = {
            "layers": layers_info,
            "total_params": total_params
        }
        
    def add_optimizer_info(self, optimizer):
        self.report_data["optimizer_info"] = {
            "type": optimizer.__class__.__name__,
            "learning_rate": getattr(optimizer, 'lr', 'N/A')
        }
        
    def add_criterion_info(self, criterion):
        self.report_data["criterion_info"] = {
            "type": criterion.__class__.__name__
        }
        
    def add_training_params(self, val_rate, epochs, batch_size, patience):
        self.report_data["training_params"] = {
            "validation_rate": val_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience
        }
        
    def add_training_history(self, history):
        self.report_data["training_history"] = history
        
    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
        
    def add_loss_accuracy_curve(self, history):
        # Loss Curve
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(history['train_loss'], label="Train Loss")
        ax1.plot(history['val_loss'], label="Val Loss")
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        # Accuracy Curve
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(history['train_acc'], label="Train Acc")
        ax2.plot(history['val_acc'], label="Val Acc")
        ax2.set_title("Accuracy Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        
        self.report_data["evaluation_results"]["loss_curve"] = self._fig_to_base64(fig1)
        self.report_data["evaluation_results"]["accuracy_curve"] = self._fig_to_base64(fig2)
        
    def add_confusion_matrix(self, labels, preds):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
        self.report_data["evaluation_results"]["confusion_matrix"] = self._fig_to_base64(fig)
        
    def add_roc_curves(self, test_labels, y_test, test_logits):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import numpy as np
        
        y_true_bin = label_binarize(test_labels, classes=np.arange(y_test.shape[1]))
        fpr, tpr, roc_auc = {}, {}, {}
        
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], test_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title('ROC Curve (One-vs-Rest)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        
        self.report_data["evaluation_results"]["roc_curves"] = self._fig_to_base64(fig)
        
    def generate_html_report(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{timestamp}.html"
            
        filepath = os.path.join(self.report_dir, filename)
        
        html_content = self._generate_html_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Report saved to: {filepath}")
        return filepath
        
    def _generate_html_content(self):
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CNN Training Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        h1 {{
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .info-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin: 15px 0;
        }}
        .timestamp {{
            color: #666;
            font-style: italic;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CNN Training Report</h1>
        <div class="timestamp">Generated on: {self.report_data["timestamp"]}</div>
        
        <h2>1. Dataset Information</h2>
        <div class="info-section">
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Training Data Shape</td>
                    <td>{self.report_data["data_info"].get("train_shape", "N/A")}</td>
                </tr>
                <tr>
                    <td>Training Labels Shape</td>
                    <td>{self.report_data["data_info"].get("train_labels_shape", "N/A")}</td>
                </tr>
                <tr>
                    <td>Test Data Shape</td>
                    <td>{self.report_data["data_info"].get("test_shape", "N/A")}</td>
                </tr>
                <tr>
                    <td>Test Labels Shape</td>
                    <td>{self.report_data["data_info"].get("test_labels_shape", "N/A")}</td>
                </tr>
                <tr>
                    <td>Dataset Level</td>
                    <td>{self.report_data["data_info"].get("dataset_level", "Custom")}</td>
                </tr>
            </table>
        </div>
        
        <h2>2. Model Architecture</h2>
        <div class="info-section">
            <p><strong>Total Parameters:</strong> {self.report_data["model_summary"].get("total_params", 0)}</p>
            <table>
                <tr>
                    <th>Layer #</th>
                    <th>Type</th>
                    <th>Input Shape</th>
                    <th>Output Shape</th>
                    <th>Parameters</th>
                    <th>Name</th>
                </tr>
        """
        
        for layer in self.report_data["model_summary"].get("layers", []):
            html += f"""
                <tr>
                    <td>{layer["index"]}</td>
                    <td>{layer["type"]}</td>
                    <td>{layer["input_shape"]}</td>
                    <td>{layer["output_shape"]}</td>
                    <td>{layer["params_count"]}</td>
                    <td>{layer["name"]}</td>
                </tr>
            """
            
        html += f"""
            </table>
        </div>
        
        <h2>3. Optimizer Configuration</h2>
        <div class="info-section">
            <table>
                <tr>
                    <th>Type</th>
                    <th>Learning Rate</th>
                </tr>
                <tr>
                    <td>{self.report_data["optimizer_info"].get("type", "N/A")}</td>
                    <td>{self.report_data["optimizer_info"].get("learning_rate", "N/A")}</td>
                </tr>
            </table>
        </div>
        
        <h2>4. Loss Function</h2>
        <div class="info-section">
            <table>
                <tr>
                    <th>Type</th>
                </tr>
                <tr>
                    <td>{self.report_data["criterion_info"].get("type", "N/A")}</td>
                </tr>
            </table>
        </div>
        
        <h2>5. Training Parameters</h2>
        <div class="info-section">
            <table>
                <tr>
                    <th>Validation Rate</th>
                    <th>Epochs</th>
                    <th>Batch Size</th>
                    <th>Patience</th>
                </tr>
                <tr>
                    <td>{self.report_data["training_params"].get("validation_rate", "N/A")}</td>
                    <td>{self.report_data["training_params"].get("epochs", "N/A")}</td>
                    <td>{self.report_data["training_params"].get("batch_size", "N/A")}</td>
                    <td>{self.report_data["training_params"].get("patience", "N/A")}</td>
                </tr>
            </table>
        </div>
        
        <h2>6. Training History</h2>
        <div class="info-section">
            <h3>Loss & Accuracy Curves</h3>
        """
        
        if "loss_curve" in self.report_data["evaluation_results"]:
            html += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{self.report_data["evaluation_results"]["loss_curve"]}" alt="Loss Curve">
            </div>
            """
            
        if "accuracy_curve" in self.report_data["evaluation_results"]:
            html += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{self.report_data["evaluation_results"]["accuracy_curve"]}" alt="Accuracy Curve">
            </div>
            """
            
        html += """
        </div>
        
        <h2>7. Evaluation Results</h2>
        <div class="info-section">
        """
        
        if "confusion_matrix" in self.report_data["evaluation_results"]:
            html += f"""
            <h3>Confusion Matrix</h3>
            <div class="image-container">
                <img src="data:image/png;base64,{self.report_data["evaluation_results"]["confusion_matrix"]}" alt="Confusion Matrix">
            </div>
            """
            
        if "roc_curves" in self.report_data["evaluation_results"]:
            html += f"""
            <h3>ROC Curves</h3>
            <div class="image-container">
                <img src="data:image/png;base64,{self.report_data["evaluation_results"]["roc_curves"]}" alt="ROC Curves">
            </div>
            """
            
        html += """
        </div>
    </div>
</body>
</html>
        """
        
        return html