import os
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def do_tsne(cfg, model, val_loader, num_query, save_path=None,
            max_samples=3000,           # 总采样上限（太大很慢）
            max_ids=20,                 # 最多画多少个ID（建议 10~30）
            samples_per_id=80,          # 每个ID最多取多少张
            pca_dim=256,                # 先PCA到低维再t-SNE更稳更快
            seed=42):
    """
    从 val_loader 抽样提特征 -> PCA -> t-SNE -> 保存散点图
    注意：你现在 model(img1,img2,img3,...) 返回 feat，假设 feat shape [B,C] 或 tuple/list 里含 [B,C]
    """
    logger = logging.getLogger("transreid.tsne")
    logger.info("Enter t-SNE visualization")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 多卡推理
    if device == "cuda" and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for inference (DataParallel)")
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    feats = []
    pids = []

    # 统计/抽样用：每个pid已经取了多少
    pid_count = {}
    selected_pids = set()

    # 先跑一遍，收集 query+gallery 的特征都可以（你 val_loader 通常就是 query+gallery 顺序拼出来的）
    with torch.no_grad():
        for n_iter, (img1, img2, img3, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            out = model(img1, img2, img3, cam_label=camids, view_label=target_view)

            # 兼容：有些实现返回 (feat, ...) 或 list/tuple
            if isinstance(out, (tuple, list)):
                feat = out[0]
            else:
                feat = out

            # 确保 [B,C]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)

            feat = feat.detach().cpu().float().numpy()
            pid_np = pid.detach().cpu().numpy() if torch.is_tensor(pid) else np.asarray(pid)

            # 抽样策略：限制 ID 数、每个ID样本数、总样本数
            for i in range(len(pid_np)):
                this_pid = int(pid_np[i])

                if this_pid not in pid_count:
                    pid_count[this_pid] = 0

                # 控制最多画 max_ids 个身份
                if this_pid not in selected_pids:
                    if len(selected_pids) >= max_ids:
                        continue
                    selected_pids.add(this_pid)

                # 控制每个ID最多取 samples_per_id
                if pid_count[this_pid] >= samples_per_id:
                    continue

                feats.append(feat[i])
                pids.append(this_pid)
                pid_count[this_pid] += 1

                if len(feats) >= max_samples:
                    break

            if len(feats) >= max_samples:
                break

    if len(feats) < 10:
        raise RuntimeError(f"Collected too few samples for t-SNE: {len(feats)}. "
                           f"Try increasing max_ids/samples_per_id or check val_loader output.")

    X = np.stack(feats, axis=0)  # [N,C]
    y = np.asarray(pids)         # [N]

    logger.info(f"Collected features: X={X.shape}, unique IDs={len(np.unique(y))}")

    # 先 PCA 再 t-SNE（强烈建议）
    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xp = pca.fit_transform(X)
    else:
        Xp = X

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(Xp) - 1) // 10)),  # 根据样本数自适应
        learning_rate="auto",
        init="pca",
        random_state=seed
    )
    Z = tsne.fit_transform(Xp)  # [N,2]

    # 画图
    plt.figure(figsize=(10, 8))
    uniq = np.unique(y)

    for pid_i in uniq:
        idx = (y == pid_i)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.8, label=str(pid_i))

    plt.title(f"t-SNE (N={len(y)}, IDs={len(uniq)})")
    plt.xticks([])
    plt.yticks([])

    # legend 太大就关掉或改成外置
    if len(uniq) <= 15:
        plt.legend(markerscale=2, fontsize=8, loc="best")
    else:
        # IDs多的时候别放legend，不然看不清
        pass

    if save_path is None:
        save_path = os.path.join(cfg.OUTPUT_DIR, "tsne.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved t-SNE figure to: {save_path}")
    return save_path