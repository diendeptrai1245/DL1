import json
import config
import matplotlib.pyplot as plt

def plot_training_history():
    """
    Loads the saved training history from a JSON file
    and generates the accuracy and loss plots.
    """
    print(f"Loading history from {config.HISTORY_SAVE_PATH}...")
    try:
        with open(config.HISTORY_SAVE_PATH, 'r') as f:
            history_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find history file at {config.HISTORY_SAVE_PATH}")
        print("Please run train.py first to generate the history file.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config.HISTORY_SAVE_PATH}.")
        return

    try:
        acc = history_data['accuracy']
        val_acc = history_data['val_accuracy']
        loss = history_data['loss']
        val_loss = history_data['val_loss']
    except KeyError as e:
        print(f"Error: History file is missing key {e}. Trying 'acc'/'val_acc'...")
        acc = history_data.get('acc', history_data.get('accuracy'))
        val_acc = history_data.get('val_acc', history_data.get('val_accuracy'))
        loss = history_data.get('loss')
        val_loss = history_data.get('val_loss')

    if not all([acc, val_acc, loss, val_loss]):
        print("Error: Could not find all necessary history keys.")
        return

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Save and show the plot
    plot_path = config.BASE_DIR / 'training_history_plot.png'
    plt.savefig(plot_path)
    print(f"Successfully saved training plot to '{plot_path}'")
    plt.show() 

if __name__ == "__main__":
    plot_training_history()