import transformers
import torch
import shutil
    

class BERTClass(torch.nn.module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 29)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    valid_loss_min = checkpoint["valid_loss_min"]
    return model, optimizer, checkpoint["epoch"], valid_loss_min


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)


def train_model(
        start_epochs,
        n_epochs,
        valid_loss_min_input,
        training_loader,
        validation_loader,
        model,
        device,
        optimizer,
        checkpoint_path,
        best_model_path
):
    # Initiialize valid loss minimum at input.
    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, n_epochs):
        train_loss = 0
        valid_loss = 0
        # Put model in training mode.
        model.train()

        print(f" -- Epoch {epoch}: Training Start -- ")
        
        for batch_idx, data in enumerate(training_loader):
            # Save batch info to device
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            # Run prediction on model for batch
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            # Evaluate loss
            loss = loss_fn(outputs, targets)
            
            if batch_idx%5000 == 0:
                print(f"Epoch: {epoch}, Training Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += (1 / (batch_idx + 1))*(loss.item() - train_loss)
        
        print(f" -- Epoch {epoch}: Training End -- ")

        print(f" -- Epoch {epoch}: Validation Start -- ")

        model.eval()

        with torch.no_grad():
            val_targets = []
            val_outputs = []
            for batch_idx, data in enumerate(training_loader):
                # Save batch info to device
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.float)
                # Evalutate model on batch
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss += (1 / (batch_idx + 1))*(loss.item() - valid_loss)
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        print(f" -- Epoch {epoch}: Validation End --")

        train_loss = train_loss/len(training_loader)
        valid_loss = valid_loss/len(validation_loader)

        print(f"Epoch: {epoch}\n\tAverage Training Loss: {train_loss}\n\tAverage Validation Loss: {valid_loss}")

        checkpoint = {
            "epoch": epoch + 1, 
            "valid_loss_min": valid_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, False, checkpoint_path, best_model_path)

        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}). Saving Model...")
            save_checkpoint(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

        print(f" -- Epoch {epoch} Done -- ")
    
    return model