import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch import loggers as pl_loggers
import torchmetrics
import numpy as np
from transformers import OneFormerProcessor, OneFormerModel, TrainingArguments, Trainer, OneFormerForUniversalSegmentation

from ..metrics import *
from ..dataloaders.datamodule import AssemblyDataModule
from training.lightning_trainers.lightning_model import LitModel


class OneFormerLitModel(LitModel):
    """
    Lightning model for finetuning OneFormer datasets.
    
    This inherits some methods from the LitModel class.
    """

    def __init__(self, learning_rate=0.001, weight_decay=0.1):
        super(OneFormerLitModel, self).__init__()

        id2label = {
            0: "background",
            1: "left_hand",
            2: "right_hand"
        }
        

        self.model = OneFormerForUniversalSegmentation.from_pretrained(
                                                                    "shi-labs/oneformer_ade20k_swin_tiny", 
                                                                    id2label=id2label, 
                                                                    label2id = {v: k for k, v in id2label.items()},
                                                                    num_classes=3, 
                                                                    ignore_mismatched_sizes=True, 
                                                                    num_queries=3
                                                                        )
        
        self.processor = OneFormerProcessor.from_pretrained(
                                                            "shi-labs/oneformer_ade20k_swin_tiny",
                                                            metadata=id2label, 
                                                            num_labels=3, 
                                                            class_names=["background", "left_hand", "right_hand"], 
                                                            do_reduce_labels=True,
                                                            size= (161 * 4, 161 * 4), # this MUST be the size of the image * 4
                                                            # do_resize=False,
                                                            ignore_mismatched_sizes=True) 
        
        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay



    def predict_step(self, batch, batch_idx):
        # used only when loading a model from a checkpoint
        # use argmax here
        x, y = batch
        inputs = self.processor(x, ["semantic"], return_tensors="pt")
        raw_preds = self.model(**inputs)
        raw_preds = raw_preds.transformer_decoder_mask_predictions
        probs = F.softmax(raw_preds, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer)
        return optimizer
    
    def _common_set(self, batch, batch_idx):
        x, y = batch
        inputs = self.processor(images=x, task_inputs=["semantic" for _ in range(x.shape[0])], return_tensors="pt")
        inputs = inputs.to("cuda")
        raw_preds = self.model(**inputs)
        loss = self.get_loss(raw_preds, y)
        


        # loss = F.cross_entropy(raw_preds, y.long())
        return loss, raw_preds.transformer_decoder_mask_predictions
    
    
    def get_loss(self, raw_preds, y):
        losses = []


        for target in y:
            ground = torch.nn.functional.one_hot(target.long(), num_classes=3)
            ground_truth = ground.permute(0,-1, 1, 2)
            loss_dict = self.model.get_loss_dict(
                masks_queries_logits=raw_preds.masks_queries_logits,
                class_queries_logits=raw_preds.class_queries_logits,
                contrastive_queries_logits=raw_preds.transformer_decoder_contrastive_queries,
                mask_labels = [ground_truth[i].to(torch.float32) for i in range(target.shape[0])],
                class_labels=[torch.tensor([0, 1, 2]).to(self.device) for i in range(ground.shape[0])], # had to remove .to(device)
                text_queries = raw_preds.text_queries,
                auxiliary_predictions= raw_preds.transformer_decoder_auxiliary_predictions,
                calculate_contrastive_loss= False
            )
            loss = self.model.get_loss(loss_dict)
            losses.append(loss)
        return sum(losses)  / 2



if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    model = OneFormerLitModel()
    dm = AssemblyDataModule(
        fit_query= ['Test_Subject_1', 'ood', 'J', 'Top_View'],
        test_query= ['Test_Subject_1', 'ood', 'TB', 'Side_View']
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./logs', name='oneformer')

    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=True,
                         profiler="pytorch")

    # trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')
    trainer.fit(model, dm)

    trainer.test(model, dm)