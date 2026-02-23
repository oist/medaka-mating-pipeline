# 跟踪数据处理模块

## 快速开始

### 一行命令处理并可视化

```bash
python scripts/tracking/main.py "d:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv" --sleap-output-dir "d:\sleap_output" --output-dir "d:\sleap_output\processed"
```

或者使用批处理文件（Windows）：

```bash
scripts\tracking\run_example.bat
```

## 功能说明

1. **自动生成file_sets.json**：自动扫描并分组同一实验的3个视角文件
2. **ID分配**：使用Hungarian算法为每条鱼分配唯一且一致的ID
3. **异常值处理**：检测并移除异常的位置数据
4. **缺失值填充**：使用插值方法填充缺失的位置数据
5. **可视化**：生成ID分配后和最终处理后的轨迹可视化

## 详细文档

请参阅 [ALGORITHM_DESCRIPTION.md](ALGORITHM_DESCRIPTION.md) 了解详细的算法说明。

## 输出文件

处理完成后，会在输出目录生成：

- `{set_key}_player1_processed.csv` - PLAYER1处理后的数据
- `{set_key}_player2_processed.csv` - PLAYER2处理后的数据
- `{set_key}_player3_processed.csv` - PLAYER3处理后的数据
- `{set_key}_after_id_assignment.png` - ID分配后的可视化
- `{set_key}_final.png` - 最终处理后的可视化
