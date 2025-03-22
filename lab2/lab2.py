import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def construct_image(U,S,V,rank=-1):
    if rank > len(S):
        rank = len(S)
    S[rank:] = 0
    sigma = np.zeros((U.shape[0], V.shape[0]))
    U[:, rank:] = 0
    V[rank:] = 0
    for i in range(rank):
        sigma[i, i] = S[i]
    new_image =U @ sigma @ V 
    new_image = np.clip(new_image, 0, 255)  
    new_image = np.uint8(new_image) # 必须要进行类型转换
    return new_image

def get_low_rank_components(U, S, V, rank):
    """获取low rank的USV矩阵"""
    if rank > len(S):
        rank = len(S)
    
    U_low = U[:, :rank].copy()
    S_low = S[:rank].copy()
    V_low = V[:rank, :].copy()
    
    return U_low, S_low, V_low

def process_image(image, rank):
    """
    处理传入的直接的BGR图像，返回重构图像和low rank的分解结果
    """
    B, G, R = cv2.split(image)
    ub, sb, vb = np.linalg.svd(B, full_matrices=False)
    ug, sg, vg = np.linalg.svd(G, full_matrices=False)
    ur, sr, vr = np.linalg.svd(R, full_matrices=False)
    
    # 获取low rank分解结果
    ub_low, sb_low, vb_low = get_low_rank_components(ub, sb, vb, rank)
    ug_low, sg_low, vg_low = get_low_rank_components(ug, sg, vg, rank)
    ur_low, sr_low, vr_low = get_low_rank_components(ur, sr, vr, rank)
    
    # 保存分解结果
    decomposition_data = {
        'B': {'U': ub_low, 'S': sb_low, 'V': vb_low},
        'G': {'U': ug_low, 'S': sg_low, 'V': vg_low},
        'R': {'U': ur_low, 'S': sr_low, 'V': vr_low}
    }
    
    new_b = construct_image(ub, sb, vb, rank)
    new_g = construct_image(ug, sg, vg, rank)
    new_r = construct_image(ur, sr, vr, rank)
    new_image = cv2.merge([new_b, new_g, new_r])
    
    return new_image, decomposition_data

def save_svd_data(decomposition_data, rank, output_dir):
    """保存SVD分解结果 (low rank)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"svd_data_rank_{rank}.npz")
    np.savez(filename, 
             ub=decomposition_data['B']['U'], sb=decomposition_data['B']['S'], vb=decomposition_data['B']['V'],
             ug=decomposition_data['G']['U'], sg=decomposition_data['G']['S'], vg=decomposition_data['G']['V'],
             ur=decomposition_data['R']['U'], sr=decomposition_data['R']['S'], vr=decomposition_data['R']['V'])
    
    return filename

def load_svd_data(filename):
    """加载SVD分解结果"""
    data = np.load(filename)
    decomposition_data = {
        'B': {'U': data['ub'], 'S': data['sb'], 'V': data['vb']},
        'G': {'U': data['ug'], 'S': data['sg'], 'V': data['vg']},
        'R': {'U': data['ur'], 'S': data['sr'], 'V': data['vr']}
    }
    return decomposition_data

def reconstruct_from_low_rank_svd(decomposition_data):
    """从low rank SVD数据重构图像"""
    # 对蓝色通道重构
    ub_low = decomposition_data['B']['U']
    sb_low = decomposition_data['B']['S']
    vb_low = decomposition_data['B']['V']
    new_b = ub_low @ np.diag(sb_low) @ vb_low
    new_b = np.clip(new_b, 0, 255).astype(np.uint8)
    
    # 对绿色通道重构
    ug_low = decomposition_data['G']['U']
    sg_low = decomposition_data['G']['S']
    vg_low = decomposition_data['G']['V']
    new_g = ug_low @ np.diag(sg_low) @ vg_low
    new_g = np.clip(new_g, 0, 255).astype(np.uint8)
    
    # 对红色通道重构
    ur_low = decomposition_data['R']['U']
    sr_low = decomposition_data['R']['S']
    vr_low = decomposition_data['R']['V']
    new_r = ur_low @ np.diag(sr_low) @ vr_low
    new_r = np.clip(new_r, 0, 255).astype(np.uint8)
    
    # 合并通道
    new_image = cv2.merge([new_b, new_g, new_r])
    return new_image

def calculate_metrics(original, reconstructed):
    """计算图像差异指标"""
    # 均方误差 (MSE)
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    
    # 峰值信噪比 (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # 结构相似性指数 (SSIM)
    # 注意：完整的SSIM需要更复杂的实现，这里使用OpenCV的实现
    # ssim = cv2.quality.QualitySSIM_compute(original, reconstructed)[0]

    from skimage.metrics import structural_similarity as ssim
    # 转换为灰度图像计算SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, reconstructed_gray)
        
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value
    }

def process_and_save_ranks(image, ranks, output_base_dir):
    """处理并保存不同rank的结果"""
    # 创建保存目录
    svd_dir = os.path.join(output_base_dir, 'SVD')
    img_dir = os.path.join(output_base_dir, 'image')
    
    if not os.path.exists(svd_dir):
        os.makedirs(svd_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    metrics_results = []
    
    for rank in ranks:
        # 处理图像
        reconstructed, decomposition_data = process_image(image, rank)
        
        # 保存SVD数据到SVD文件夹
        svd_filename = save_svd_data(decomposition_data, rank, svd_dir)
        
        # 保存重构图像到image文件夹
        img_filename = os.path.join(img_dir, f"reconstructed_rank_{rank}.jpg")
        cv2.imwrite(img_filename, reconstructed)
        
        # 从保存的SVD数据重建图像
        loaded_decomposition_data = load_svd_data(svd_filename)
        reconstructed_from_saved = reconstruct_from_low_rank_svd(loaded_decomposition_data)
        
        # 保存从SVD数据重建的图像
        recon_img_filename = os.path.join(img_dir, f"from_saved_svd_rank_{rank}.jpg")
        cv2.imwrite(recon_img_filename, reconstructed_from_saved)
        
        # 计算和存储指标 (对比原图和重构图像)
        metrics = calculate_metrics(image, reconstructed)
        metrics['rank'] = rank
        metrics_results.append(metrics)
        
        # 计算保存的SVD数据重建图像与原图的对比指标
        metrics_from_saved = calculate_metrics(image, reconstructed_from_saved)
        
        print(f"处理完成 rank={rank}")
        print(f"  直接重构: MSE={metrics['MSE']:.2f}, PSNR={metrics['PSNR']:.2f} dB, SSIM={metrics['SSIM']:.4f}")
        print(f"  SVD重建: MSE={metrics_from_saved['MSE']:.2f}, PSNR={metrics_from_saved['PSNR']:.2f} dB, SSIM={metrics_from_saved['SSIM']:.4f}")
    
    # 生成比较图表并保存到image文件夹
    plot_metrics(metrics_results, img_dir)
    
    return metrics_results

def plot_metrics(metrics_results, output_dir):
    """绘制并保存指标比较图表"""
    ranks = [m['rank'] for m in metrics_results]
    mse_values = [m['MSE'] for m in metrics_results]
    psnr_values = [m['PSNR'] for m in metrics_results]
    ssim_values = [m['SSIM'] for m in metrics_results]
    
    # 创建图表
    plt.figure(figsize=(15, 5))
    
    # MSE图
    plt.subplot(1, 3, 1)
    plt.plot(ranks, mse_values, 'b-o')
    plt.title('均方误差 (MSE)')
    plt.xlabel('Rank')
    plt.ylabel('MSE')
    plt.grid(True)
    
    # PSNR图
    plt.subplot(1, 3, 2)
    plt.plot(ranks, psnr_values, 'g-o')
    plt.title('峰值信噪比 (PSNR)')
    plt.xlabel('Rank')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    # SSIM图
    plt.subplot(1, 3, 3)
    plt.plot(ranks, ssim_values, 'r-o')
    plt.title('结构相似性 (SSIM)')
    plt.xlabel('Rank')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_metrics.png'))
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
    os.chdir(script_dir)  # 把当前工作目录改为脚本所在目录
    
    # 加载图像
    image = cv2.imread("fruits.jpg")
    if image is None:
        print("无法加载图像文件")
        return
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, 'results')
    
    # 处理不同的rank值
    ranks = [1, 5, 10, 30, 100,300]
    metrics_results = process_and_save_ranks(image, ranks, output_dir)
    
    # 打印比较结果
    print("\n结果对比:")
    print("Rank\tMSE\t\tPSNR\t\tSSIM")
    for m in metrics_results:
        print(f"{m['rank']}\t{m['MSE']:.2f}\t\t{m['PSNR']:.2f} dB\t{m['SSIM']:.4f}")
    
    # 展示原图和两个重构图像(直接重构和从保存的SVD重构)
    rank_to_show = 100
    direct_reconstructed = cv2.imread(os.path.join(output_dir, 'image', f"reconstructed_rank_{rank_to_show}.jpg"))
    svd_reconstructed = cv2.imread(os.path.join(output_dir, 'image', f"from_saved_svd_rank_{rank_to_show}.jpg"))
    
    # 显示原图和重构图像
    cv2.imshow("Original", image)
    cv2.imshow(f"Direct Reconstructed (Rank {rank_to_show})", direct_reconstructed)
    cv2.imshow(f"From Saved SVD (Rank {rank_to_show})", svd_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


