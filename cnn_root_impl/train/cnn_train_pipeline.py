import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from cnn_root_impl.util.report_generator import ReportGenerator


class CNN_Training:

    def __init__(self, model, optimizer, criterion,
                 x_train, y_train,
                 x_test, y_test,
                 val_rate=0.2,
                 epochs=20, batch_size=32,
                 patience=3, report=False, dataset_level=None):
        # Split training data into train and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=val_rate, random_state=42
        )

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.report = report
        self.dataset_level = dataset_level
        self.val_rate = val_rate

    def train(self):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        x_train = self.x_train
        y_train = self.y_train
        x_val = self.x_val
        y_val = self.y_val
        epochs = self.epochs
        batch_size = self.batch_size
        patience = self.patience

        num_train = x_train.shape[0]
        num_batches = num_train // batch_size

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        best_val_loss = np.inf
        best_epoch = 0
        best_weights = None

        print(f"\n\nTraining Parameters: ")
        print("-" * 50)
        print(f"epochs: {epochs}")
        print(f"patient: {patience}")
        print(f"batch_size: {batch_size}")
        print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}")

        print(f"\n\nStart Training: ")
        print("-" * 50)
        for epoch in range(epochs):
            # ---- Shuffle ----
            indices = np.arange(num_train)
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            epoch_loss, correct = 0.0, 0
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for i in pbar:
                start, end = i * batch_size, (i + 1) * batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]

                # Forward
                logits = model.forward(x_batch, training=True, verbose=False)
                loss = criterion.forward(logits, y_batch)
                epoch_loss += loss

                preds = np.argmax(logits, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct += np.sum(preds == labels)

                # Backward + Update
                grad = criterion.backward()
                model.backward(grad, verbose=False)
                optimizer.step(model)

                pbar.set_postfix(loss=loss)

            avg_loss = epoch_loss / num_batches
            train_acc = correct / (num_batches * batch_size)

            # ---- Validation ----
            val_logits = model.forward(x_val, training=False, verbose=False)
            val_loss = criterion.forward(val_logits, y_val)
            val_preds = np.argmax(val_logits, axis=1)
            val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_preds == val_labels)

            train_losses.append(avg_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}: "
                  f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")

            # ---- EarlyStopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = model
            elif epoch - best_epoch >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        print("\nTraining Completed")
        print("-" * 50)

        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs
        }
        
        # Generate report if requested
        if self.report:
            report_gen = ReportGenerator()
            report_gen.add_data_info(
                x_train.shape, y_train.shape, 
                self.x_test.shape, self.y_test.shape,
                self.dataset_level
            )
            report_gen.add_model_summary(model)
            report_gen.add_optimizer_info(optimizer)
            report_gen.add_criterion_info(criterion)
            report_gen.add_training_params(val_rate=self.val_rate, 
                                          epochs=epochs, batch_size=batch_size, patience=patience)
            
            return history, report_gen
            
        return history