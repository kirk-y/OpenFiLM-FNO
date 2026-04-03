import json
import time
import yaml
from data.data import FSCdataset, ProcessedSampleDataset
from data.data_v2 import FSCdatasetV2
from data.data_v3 import FSCdatasetV3, SamplePathDataset, filter_sample_paths
from torch.utils.data import DataLoader
from training.trainer import FluidSolidTrainer
from models.fno import FluidSolidFNOmodel
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from common.visualization import draw_all, plot_metric_series
import os
from pathlib import Path
from datetime import datetime
from common.loss import loss_fn
import pickle

def prepare_datasets(config):
    data_cfg = config['data']
    origin_path = data_cfg['origin_path']
    data_version = str(data_cfg.get('version', '')).lower()
    use_v3 = data_version == 'v3'
    use_v2 = bool(data_cfg.get('use_v2')) or data_version == 'v2'
    if not use_v2 and not use_v3:
        try:
            if 'hfsdata_v2' in str(origin_path) or Path(origin_path).name == 'samples':
                use_v2 = True
        except Exception:
            use_v2 = False

    if use_v3:
        v3_rates = data_cfg.get('rates', [0.7, 0.2, 0.1])
        fsc = FSCdatasetV3(
            max_n=int(data_cfg['max_n']),
            file_path=origin_path,
            save_path=data_cfg['save_path'],
            rates=v3_rates,
            seed=int(data_cfg['seed']),
            max_conditions=data_cfg.get('max_conditions'),
            max_samples_per_condition=data_cfg.get('max_samples_per_condition'),
            skip_invalid_samples=bool(data_cfg.get('skip_invalid_samples', False)),
            export_stats=bool(data_cfg.get('export_stats', False)),
            stats_format=str(data_cfg.get('stats_format', 'yaml')),
            stats_filename=str(data_cfg.get('stats_filename', 'dataset_stats.yaml')),
        )
    elif use_v2:
        v2_rates = [0.7, 0.2, 0.1]
        fsc = FSCdatasetV2(max_n=int(data_cfg['max_n']),
                           file_path=origin_path,
                           save_path=data_cfg['save_path'],
                           rates=v2_rates,
                           seed=int(data_cfg['seed']),
                           max_conditions=data_cfg.get('max_conditions'),
                           use_preprocessed=bool(data_cfg.get('v2_use_preprocessed', True)))
    else:
        fsc = FSCdataset(max_n=int(data_cfg['max_n']),
                         file_path=origin_path,
                         save_path=data_cfg['save_path'],
                         noise_level=float(data_cfg['noise_level']),
                         rates=data_cfg['rates'],
                         seed=int(data_cfg['seed']))
    
    testing_cfg = config.get('testing', {})
    test_data_path = testing_cfg.get('test_data_path')
    
    # Check/Generate Test Data
    if test_data_path and (not use_v2) and (not use_v3):
        if not os.path.exists(test_data_path):
            save_dir = os.path.dirname(test_data_path)
            json_path = os.path.join(save_dir, 'test_paths.json')
            
            if os.path.exists(json_path):
                print(f"Found test paths JSON at {json_path}. Skipping generation.")
            else:
                print(f"Test data not found at {test_data_path} or {json_path}. Attempting to generate...")
                raw_test_dir = 'data/hfsdata/testset/'
                stats_path = 'data/hfsdata/processed_data3_x_x/stats.json'
                
                if os.path.exists(raw_test_dir) and os.path.exists(stats_path):
                     print(f"Generating test data from {raw_test_dir} using stats from {stats_path}")
                     temp_fsc = FSCdataset(max_n=2500, file_path=raw_test_dir, save_path=save_dir, rates=[0, 0, 1])
                     temp_fsc.init_local_data(stats_source=stats_path)
                else:
                     print(f"Cannot generate test data: raw dir {raw_test_dir} or stats {stats_path} missing.")

    def has_index_files(path: str) -> bool:
        return (
            os.path.exists(os.path.join(path, 'train_paths.json'))
            and os.path.exists(os.path.join(path, 'val_paths.json'))
            and os.path.exists(os.path.join(path, 'test_paths.json'))
        )

    index_root = None
    save_path = data_cfg.get('save_path')
    if save_path and os.path.exists(save_path) and has_index_files(save_path):
        index_root = save_path
        print(f"检测到预处理索引: {save_path}")
    elif os.path.exists(origin_path) and has_index_files(origin_path):
        index_root = origin_path
        print(f"检测到原始目录已预处理: {origin_path}")

    if index_root:
        train_dataset, val_dataset, default_test_dataset = fsc.load_data(index_root)
    else:
        print("未检测到预处理索引，将基于原始数据生成索引...")
        train_dataset, val_dataset, default_test_dataset = fsc.init_local_data()
    
    test_dataset = default_test_dataset
    
    if test_data_path:
        test_data_path = os.path.expanduser(str(test_data_path))
        if use_v3:
            if not os.path.isfile(test_data_path):
                raise FileNotFoundError(f"V3 测试集路径不存在: {test_data_path}")
            with open(test_data_path, 'r', encoding='utf-8') as f:
                paths = json.load(f)
            paths = filter_sample_paths(
                paths,
                max_samples_per_condition=data_cfg.get('max_samples_per_condition'),
                max_conditions=data_cfg.get('max_conditions')
            )
            test_dataset = SamplePathDataset(
                paths,
                max_n=int(data_cfg['max_n']),
                preload=False,
                load_cb=bool(testing_cfg.get('load_cb_in_test', False)),
                load_ddm_info=bool(testing_cfg.get('load_ddm_in_test', False)),
                skip_invalid_samples=bool(data_cfg.get('skip_invalid_samples', False)),
            )
            print(f'V3 自定义测试集已加载: {test_data_path}')
            print(f"测试集样本数: {len(test_dataset)}")
        elif use_v2:
            if os.path.exists(test_data_path):
                print(f"Loading test set from {test_data_path}")
                with open(test_data_path, 'rb') as f:
                    test_dataset = pickle.load(f)
            else:
                save_dir = os.path.dirname(test_data_path)
                json_path = os.path.join(save_dir, 'test_paths.json')
                if os.path.exists(json_path):
                    print(f"Loading test set from JSON paths: {json_path}")
                    with open(json_path, 'r') as f:
                        paths = json.load(f)
                    test_dataset = ProcessedSampleDataset(paths)
                    print(f'自定义测试集已从JSON路径加载: {json_path}')
                    print(f"测试集样本数: {len(test_dataset)}")
        else:
            if os.path.exists(test_data_path):
                print(f"Loading test set from {test_data_path}")
                with open(test_data_path, 'rb') as f:
                    test_dataset = pickle.load(f)
            else:
                save_dir = os.path.dirname(test_data_path)
                json_path = os.path.join(save_dir, 'test_paths.json')
                if os.path.exists(json_path):
                    print(f"Loading test set from JSON paths: {json_path}")
                    with open(json_path, 'r') as f:
                        paths = json.load(f)
                    test_dataset = ProcessedSampleDataset(paths)
                    print(f'自定义测试集已从JSON路径加载: {json_path}')
                    print(f"测试集样本数: {len(test_dataset)}")

    
    return train_dataset, val_dataset, test_dataset, fsc.collate_fn

def main(config):
    train_dataset, val_dataset, test_dataset, collate = prepare_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=collate)

    device_str = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print('检测到CUDA不可用，自动切换至CPU运行。')
        device_str = 'cpu'
    device = torch.device(device_str)
    config['training']['device'] = device_str

    model = FluidSolidFNOmodel(config)
    model.to(device)
    param_num = sum(p.numel() for p in model.parameters())

    train_enabled = bool(config['training'].get('enable_training', True))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = Path(config['training']['exp_path'])
    exp_root.mkdir(parents=True, exist_ok=True)
    if train_enabled:
        exp_dir = exp_root / f'fsc_exp_name_{model.name()}_{timestamp}_pramaNum_{param_num}'
    else:
        default_eval_dir = exp_root / f'fsc_eval_{timestamp}'
        exp_dir = Path(config.get('testing', {}).get('exp_dir', default_eval_dir))
    exp_dir.mkdir(parents=True, exist_ok=True)

    dir_label = '实验目录' if train_enabled else '测试输出目录'
    print(f"模型参数总数: {param_num}")
    print(f"{dir_label}: {exp_dir}")

    with open(exp_dir / 'config.yml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file)
        print(f"配置信息已保存到 {exp_dir / 'config.yml'}")

    optimizer = AdamW(model.parameters(),
                      lr=float(config['training']['lr']),
                      weight_decay=float(config['training']['weight_decay']))
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=config['training']['T_0'],
                                            T_mult=config['training']['T_mult'])
    criterion = loss_fn(config['training']['loss_type'])

    trainer = FluidSolidTrainer(config, model, optimizer, scheduler, criterion,
                                train_loader, val_loader, test_loader,
                                device, exp_dir)

    if train_enabled:
        trainer.train()
    else:
        print('跳过训练，仅执行测试流程。')

    testing_cfg = config.get('testing', {})
    if not testing_cfg.get('enabled', False):
        return

    checkpoint_path = testing_cfg.get('checkpoint_path')
    if not checkpoint_path:
        if train_enabled and trainer.best_checkpoint:
            checkpoint_path = trainer.best_checkpoint
        else:
            checkpoint_path = config.get('model', {}).get('save_path')

    if not checkpoint_path:
        print('测试被跳过: 未提供可用的模型权重路径。')
        return

    checkpoint_path = os.path.expanduser(str(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        print(f'测试被跳过: 未找到模型权重 {checkpoint_path}')
        return

    trainer.load_checkpoint(checkpoint_path)
    trainer.model.to(device)

    max_samples_cfg = testing_cfg.get('max_samples')
    max_samples = None
    if max_samples_cfg not in (None, 'all'):
        try:
            max_samples = int(max_samples_cfg)
        except (TypeError, ValueError):
            print(f'无效的 max_samples 设置 {max_samples_cfg}，将处理全部测试样本。')

    start_time = time.time()
    preds, targets, w_old, fi, ns, metrics, per_sample_metrics = trainer.test(max_samples=max_samples)
    elapsed = time.time() - start_time
    print(f"测试耗时: {elapsed:.2f} 秒")

    if not preds:
        print('测试数据为空，未生成可视化图表。')
        return

    # 导出最大规模样本的预测和标签数据
    try:
        import numpy as np
        import csv
        max_n_idx = np.argmax(ns)
        max_n_val = ns[max_n_idx]
        print(f"最大规模样本索引: {max_n_idx}, 节点数: {max_n_val}")

        pred_max = preds[max_n_idx]
        target_max = targets[max_n_idx]
        w_old_max = w_old[max_n_idx]

        csv_filename = f'max_size_prediction_target_n{max_n_val}.csv'
        csv_path = exp_dir / csv_filename
        
        # 确保数据是一维的
        if pred_max.ndim > 1:
            pred_max = pred_max.flatten()
        if target_max.ndim > 1:
            target_max = target_max.flatten()
        if w_old_max.ndim > 1:
            w_old_max = w_old_max.flatten()

        # 计算总宽度 (w_old + increment)
        total_pred = w_old_max + pred_max
        total_target = target_max

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Node_Index', 'Prediction', 'Target'])
            for i in range(len(total_pred)):
                writer.writerow([i, total_pred[i], total_target[i]])
        print(f"最大规模样本预测与标签数据已保存至: {csv_path}")
    except Exception as e:
        print(f"保存最大规模样本数据时出错: {e}")

    figures_dir = Path(testing_cfg.get('figures_dir', exp_dir / 'test_figures'))
    conditions = per_sample_metrics.get('condition', [])
    top_indices = []
    if conditions:
        from collections import defaultdict
        cond_to_indices = defaultdict(list)
        for idx, cond in enumerate(conditions):
            cond_to_indices[cond].append(idx)
        for cond, indices in cond_to_indices.items():
            top_indices.extend(indices[-20:])
    draw_all(preds, targets, w_old, ns, fi, figures_dir, sample_indices=top_indices or None)
    plot_metric_series(per_sample_metrics.get('relative_l2', []), 'relative_l2_error', figures_dir)
    plot_metric_series(per_sample_metrics.get('nrmse', []), 'nrmse', figures_dir)
    print(f"测试样本数: {len(preds)}")
    print(f"测试指标: {metrics}")
    print(f"可视化图表保存至: {figures_dir}")

    metrics_path = Path(testing_cfg.get('metrics_path', figures_dir / 'metrics.json'))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as metric_file:
        json.dump(metrics, metric_file, indent=2, ensure_ascii=False)
    print(f"指标文件保存至: {metrics_path}")

    per_sample_metrics_path = metrics_path.with_name(metrics_path.stem + '_per_sample.json')
    with open(per_sample_metrics_path, 'w', encoding='utf-8') as per_sample_file:
        json.dump(per_sample_metrics, per_sample_file, indent=2, ensure_ascii=False)
    print(f"逐样本指标保存至: {per_sample_metrics_path}")

    from common.bark import send
    send("FSI FNOmodel Message","模型训练完成",
              token="kirkyang", sender="FSI_FNOmodel")




if __name__ == '__main__':
    # 打开YAML文件并加载内容
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        # print(config)
    main(config)