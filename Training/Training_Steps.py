import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          model_save_folder: str,
          epochs: int,
          device):

    # Initialize results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Set up the plot figure
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Initial plot settings for both subplots
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")

    # Initialize line objects for training and test metrics
    train_loss_line, = ax1.plot([], [], label="train_loss", color="blue")
    test_loss_line, = ax1.plot([], [], label="test_loss", color="orange")
    train_acc_line, = ax2.plot([], [], label="train_acc", color="blue")
    test_acc_line, = ax2.plot([], [], label="test_acc", color="orange")

    # Add legends
    ax1.legend()
    ax2.legend()

    plt.show()

    for epoch in tqdm(range(epochs)):

        start_time = timer()

        # Perform a training step
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        # Perform a test/validation step
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        scheduler.step(test_loss)

        end_time = timer()

        # Save the model
        if epoch % 5 == 0 and epoch != 0:
            if model_save_folder is not None:
                torch.save(model.state_dict(), model_save_folder + f'/model_{epoch}.pth')

        # Print the progress for each epoch
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | "
              f"Total training time: {end_time - start_time:.3f} seconds")

        # Append results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Update line data
        epochs_range = range(1, epoch + 2)
        train_loss_line.set_data(epochs_range, results["train_loss"])
        test_loss_line.set_data(epochs_range, results["test_loss"])
        train_acc_line.set_data(epochs_range, results["train_acc"])
        test_acc_line.set_data(epochs_range, results["test_acc"])

        # Set plot limits dynamically based on data
        ax1.set_xlim(1, epochs)
        ax1.set_ylim(0, max(max(results["train_loss"], default=0), max(results["test_loss"], default=0)) * 1.1)
        ax2.set_xlim(1, epochs)
        ax2.set_ylim(0, 1.1)  # Assuming accuracy is between 0 and 1

        # Redraw the figure
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot

    return results