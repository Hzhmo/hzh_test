from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 固定种子，确保每次训练结果一致

# ========== 1. 定义带软标签的损失函数 ==========
class SoftLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, soft_labels):
        # soft_labels 是 [batch_size, num_classes] 的概率分布
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(soft_labels * log_probs, dim=-1).mean()
        return loss


# ========== 2. 读取带软标签的数据 ==========
# 在 train_soft.py 的读取数据部分替换为：

# ========== 2. 读取带软标签的数据 ==========
df = pd.read_csv("data/train_with_soft_labels.csv")

# 🔥 清理重复的标题行
# 删除那些 text 列等于 "text" 的行（重复的标题行）
df = df[df['text'] != 'text'].copy()

# 确保 label 和 soft_label 是数字类型
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df['soft_label'] = pd.to_numeric(df['soft_label'], errors='coerce')

# 删除转换失败的行
df = df.dropna(subset=['label', 'soft_label'])

print(f"数据量（清理后）: {len(df)}")
print(f"\n数据分布:")
print(f"正常短信(0): {len(df[df['label']==0])} 条")
print(f"勒索短信(1): {len(df[df['label']==1])} 条")
print(f"广告短信(2): {len(df[df['label']==2])} 条")

# 验证 soft_label 范围
print(f"\nsoft_label 统计:")
print(f"  勒索类: min={df[df['label']==1]['soft_label'].min():.2f}, max={df[df['label']==1]['soft_label'].max():.2f}")
print(f"  正常类: min={df[df['label']==0]['soft_label'].min():.2f}, max={df[df['label']==0]['soft_label'].max():.2f}")
print(f"  广告类: min={df[df['label']==2]['soft_label'].min():.2f}, max={df[df['label']==2]['soft_label'].max():.2f}")


# 🔥 修复后的分布生成函数
def create_distribution(row):
    label = int(row['label'])
    soft = float(row['soft_label'])

    if label == 1:  # 勒索短信
        if soft > 0.8:
            return [0.02, 0.93, 0.05]
        else:
            return [0.05, 0.85, 0.10]
    elif label == 2:  # 广告短信
        if soft > 0.3:
            return [0.05, 0.05, 0.90]
        elif soft > 0.1:
            return [0.10, 0.10, 0.80]
        else:
            return [0.15, 0.10, 0.75]
    else:  # 正常短信 (label == 0)
        if soft > 0.3:
            return [0.90, 0.05, 0.05]
        elif soft > 0.1:
            return [0.85, 0.08, 0.07]
        else:
            return [0.95, 0.03, 0.02]


df['soft_distribution'] = df.apply(create_distribution, axis=1)

# 打印数据分布
print("\n数据分布:")
print(f"正常短信(0): {len(df[df['label'] == 0])} 条")
print(f"勒索短信(1): {len(df[df['label'] == 1])} 条")
print(f"广告短信(2): {len(df[df['label'] == 2])} 条")

# 划分数据集（分层采样，保证每类都有代表）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"\n数据集划分:")
print(f"训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")
print(f"训练集勒索短信: {len(train_df[train_df['label'] == 1])} 条")
print(f"验证集勒索短信: {len(val_df[val_df['label'] == 1])} 条")
print(f"测试集勒索短信: {len(test_df[test_df['label'] == 1])} 条")

# ========== 3. 加载模型和分词器 ==========
model_path = r"D:\develop\bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

print(f"\n模型加载完成: {model_path}")


# ========== 4. 准备数据的函数 ==========
def prepare_data(df, tokenizer, max_length=128):
    """将 DataFrame 转换为 Dataset 格式"""
    texts = df['text'].tolist()
    soft_labels = df['soft_distribution'].tolist()

    # 对文本进行编码
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # 创建 Dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'soft_labels': soft_labels
    })

    return dataset


# ========== 5. 创建数据集 ==========
train_dataset = prepare_data(train_df, tokenizer)
val_dataset = prepare_data(val_df, tokenizer)
test_dataset = prepare_data(test_df, tokenizer)

print(f"\n数据集格式检查:")
print(f"训练集第一个样本的键: {train_dataset[0].keys()}")
print(f"soft_labels 是否存在: {'soft_labels' in train_dataset[0]}")


# ========== 6. 自定义 DataCollator ==========
class SoftLabelDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # 提取 soft_labels
        soft_labels = torch.tensor([f.pop('soft_labels') for f in features], dtype=torch.float32)

        # 使用默认的 collator 处理模型输入
        batch = self.default_collator(features)

        # 添加 soft_labels
        batch['soft_labels'] = soft_labels

        return batch


# ========== 7. 自定义Trainer（安全优先）==========
class SecureSoftLabelTrainer(Trainer):
    """
    安全优先的训练器：
    1. 支持软标签训练
    2. 对勒索类的错误预测额外惩罚
    3. 正确传递软标签用于评估
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("soft_labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 基础软标签损失
        loss_fct = SoftLabelLoss()
        base_loss = loss_fct(logits, labels)

        # 额外惩罚：对勒索类的错误预测加重惩罚
        extortion_idx = 1  # 勒索类别索引
        hard_labels = torch.argmax(labels, dim=-1)
        extortion_mask = (hard_labels == extortion_idx)

        if extortion_mask.sum() > 0:
            # 对勒索样本单独计算损失
            extortion_logits = logits[extortion_mask]
            extortion_labels = labels[extortion_mask]
            extortion_loss = loss_fct(extortion_logits, extortion_labels)
            # 总损失 = 基础损失 + 勒索类额外损失（权重x2）
            loss = base_loss + extortion_loss
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 在预测前移除 soft_labels，避免传给模型
        labels = inputs.pop("soft_labels", None)

        # 调用父类的 prediction_step
        loss, logits, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        # 如果有软标签，计算自定义损失
        if labels is not None and not prediction_loss_only:
            loss_fct = SoftLabelLoss()
            loss = loss_fct(logits, labels)

        return (loss, logits, labels)


# ========== 8. 定义安全优先的评估指标 ==========
def compute_metrics(eval_pred):
    """
    安全优先的评估指标：
    - 主要：勒索短信召回率（最重要的指标）
    - 辅助：勒索短信精确率、F1分数
    - 参考：整体准确率、各类别F1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    hard_labels = np.argmax(labels, axis=-1)

    # 类别定义
    extortion_label = 1  # 勒索短信
    normal_label = 0  # 正常短信
    ad_label = 2  # 广告短信

    # 1. 勒索短信召回率（核心指标：不能漏掉勒索短信）
    extortion_mask = (hard_labels == extortion_label)
    if extortion_mask.sum() > 0:
        extortion_recall = accuracy_score(
            hard_labels[extortion_mask],
            predictions[extortion_mask]
        )
    else:
        extortion_recall = 0.0

    # 2. 勒索短信精确率（检测为勒索的样本中真正是勒索的比例）
    pred_extortion_mask = (predictions == extortion_label)
    if pred_extortion_mask.sum() > 0:
        extortion_precision = accuracy_score(
            hard_labels[pred_extortion_mask],
            predictions[pred_extortion_mask]
        )
    else:
        extortion_precision = 0.0

    # 3. 勒索短信F1（召回和精确的平衡）
    if extortion_recall + extortion_precision > 0:
        extortion_f1 = 2 * (extortion_recall * extortion_precision) / \
                       (extortion_recall + extortion_precision)
    else:
        extortion_f1 = 0.0

    # 4. 整体性能指标（参考）
    overall_accuracy = accuracy_score(hard_labels, predictions)
    overall_f1 = f1_score(hard_labels, predictions, average='weighted')

    # 5. 每类的F1分数
    f1_per_class = f1_score(hard_labels, predictions, average=None)

    metrics = {
        # 核心指标（最重要的）
        'extortion_recall': extortion_recall,  # 目标：≥95%
        'extortion_precision': extortion_precision,  # 目标：≥70%
        'extortion_f1': extortion_f1,  # 目标：≥80%

        # 整体指标（参考）
        'accuracy': overall_accuracy,  # 目标：≥85%
        'f1_weighted': overall_f1,  # 目标：≥83%

        # 各类别F1（调试用）
        'f1_normal': f1_per_class[normal_label] if len(f1_per_class) > normal_label else 0,
        'f1_extortion': f1_per_class[extortion_label] if len(f1_per_class) > extortion_label else 0,
        'f1_ad': f1_per_class[ad_label] if len(f1_per_class) > ad_label else 0,
    }

    return metrics


# ========== 9. 训练参数（安全优先配置）==========
training_args = TrainingArguments(
    output_dir="./results_security",
    num_train_epochs=3,  # 多训练几个epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="steps",  # 更频繁的评估
    eval_steps=50,  # 每50步评估一次
    save_strategy="steps",  # 更频繁的保存
    save_steps=50,
    logging_dir="./logs_security",
    logging_steps=25,

    # 🔥 核心：用勒索短信召回率选择最佳模型
    metric_for_best_model="eval_extortion_recall",  # 最重要！
    greater_is_better=True,  # 召回率越高越好

    # 保存策略
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    save_total_limit=5,  # 保存多个检查点

    # 学习率
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,

    # 其他
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # 避免警告
    seed=42,
)

# ========== 10. 创建Trainer ==========
data_collator = SoftLabelDataCollator(tokenizer)

trainer = SecureSoftLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========== 11. 训练前检查 ==========
print("\n" + "=" * 50)
print("🚀 安全优先的勒索短信检测训练")
print("=" * 50)
print(f"主要优化指标: 勒索短信召回率 (extortion_recall)")
print(f"目标: ≥95%")
print(f"训练批次: {len(train_dataset) // 8} 步/epoch")
print(f"总训练步数: {len(train_dataset) // 8 * 5}")
print("=" * 50 + "\n")

# ========== 12. 开始训练 ==========
trainer.train()

# ========== 13. 最终评估 ==========
print("\n" + "=" * 50)
print("📊 测试集最终评估")
print("=" * 50)

test_results = trainer.evaluate(test_dataset)
for key, value in test_results.items():
    if 'extortion' in key.lower():
        print(f"🔥 {key}: {value:.4f}")
    else:
        print(f"   {key}: {value:.4f}")

# ========== 14. 安全性能分析 ==========
print("\n" + "=" * 50)
print("🛡️ 安全性能分析")
print("=" * 50)

extortion_recall = test_results.get('eval_extortion_recall', 0)
extortion_precision = test_results.get('eval_extortion_precision', 0)

if extortion_recall >= 0.95:
    print("✅ 勒索短信召回率达标 (≥95%)")
elif extortion_recall >= 0.90:
    print("⚠️ 勒索短信召回率接近目标 (90-95%)")
    print("   建议：增加勒索短信训练样本或调高损失权重")
else:
    print("❌ 勒索短信召回率不达标 (<90%)，建议：")
    print("   1. 增加勒索短信训练样本")
    print("   2. 调高勒索类损失权重")
    print("   3. 使用数据增强")

if extortion_precision >= 0.70:
    print("✅ 勒索短信精确率合理 (≥70%)")
else:
    print("⚠️ 误报率较高，可能需要调整阈值")

# ========== 15. 保存最终模型 ==========
save_path = "./model_ransom_2600aug"
os.makedirs(save_path, exist_ok=True)

# 保存模型和分词器
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 保存训练配置
config = {
    'model_type': 'bert-base-chinese',
    'num_classes': 3,
    'class_names': ['正常短信', '勒索短信', '广告短信'],
    'dataset': 'train_with_soft_labels_aug (v8: paraphrase+cross-scenario+adversarial)',
    'dataset_size': len(df),
    'primary_metric': 'extortion_recall',
    'target_recall': 0.95,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_results': {
        'extortion_recall': float(extortion_recall),
        'extortion_precision': float(extortion_precision),
        'accuracy': float(test_results.get('eval_accuracy', 0))
    }
}
with open(f"{save_path}/config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"\n✅ 安全检测模型已保存到: {save_path}")
print(f"\n使用示例:")
print(f"  from transformers import pipeline")
print(f"  detector = pipeline('text-classification', model='{save_path}')")
print(f"  result = detector('我是xx黑客，你的账号已被入侵，需要支付5000元解锁')")