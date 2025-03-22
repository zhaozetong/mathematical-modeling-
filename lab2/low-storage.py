import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False  

def process_image_yuv(image, y_rank, uv_rank):
    """
    使用YUV空间进行SVD分解，对Y通道使用较高rank，对U/V通道使用较低rank
    """
    # 转换BGR图像到YUV空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv_image)
    
    # 对Y通道进行SVD分解并截断
    uy, sy, vy = np.linalg.svd(Y.astype(np.float32), full_matrices=False)
    uy_low = uy[:, :y_rank].copy().astype(np.float32)
    sy_low = sy[:y_rank].copy().astype(np.float32)
    vy_low = vy[:y_rank, :].copy().astype(np.float32)
    
    # 对U通道进行SVD分解并截断
    uu, su, vu = np.linalg.svd(U.astype(np.float32), full_matrices=False)
    uu_low = uu[:, :uv_rank].copy().astype(np.float32)
    su_low = su[:uv_rank].copy().astype(np.float32)
    vu_low = vu[:uv_rank, :].copy().astype(np.float32)
    
    # 对V通道进行SVD分解并截断
    uv, sv, vv = np.linalg.svd(V.astype(np.float32), full_matrices=False)
    uv_low = uv[:, :uv_rank].copy().astype(np.float32)
    sv_low = sv[:uv_rank].copy().astype(np.float32)
    vv_low = vv[:uv_rank, :].copy().astype(np.float32)
    
    # 保存分解结果
    decomposition_data = {
        'Y': {'U': uy_low, 'S': sy_low, 'V': vy_low},
        'U': {'U': uu_low, 'S': su_low, 'V': vu_low},
        'V': {'U': uv_low, 'S': sv_low, 'V': vv_low}
    }
    
    return decomposition_data

def reconstruct_from_yuv_svd(decomposition_data):
    """从YUV空间的SVD数据重构图像"""
    # 重构Y通道
    uy_low = decomposition_data['Y']['U']
    sy_low = decomposition_data['Y']['S']
    vy_low = decomposition_data['Y']['V']
    
    # 创建正确维度的对角矩阵
    sy_diag = np.zeros((uy_low.shape[1], vy_low.shape[0]), dtype=np.float32)
    for i in range(len(sy_low)):
        sy_diag[i, i] = sy_low[i]
    
    new_y = uy_low @ sy_diag @ vy_low
    new_y = np.clip(new_y, 0, 255).astype(np.uint8)
    
    # 重构U通道
    uu_low = decomposition_data['U']['U']
    su_low = decomposition_data['U']['S']
    vu_low = decomposition_data['U']['V']
    
    # 创建正确维度的对角矩阵
    su_diag = np.zeros((uu_low.shape[1], vu_low.shape[0]), dtype=np.float32)
    for i in range(len(su_low)):
        su_diag[i, i] = su_low[i]
    
    new_u = uu_low @ su_diag @ vu_low
    new_u = np.clip(new_u, 0, 255).astype(np.uint8)
    
    # 重构V通道
    uv_low = decomposition_data['V']['U']
    sv_low = decomposition_data['V']['S']
    vv_low = decomposition_data['V']['V']
    
    # 创建正确维度的对角矩阵
    sv_diag = np.zeros((uv_low.shape[1], vv_low.shape[0]), dtype=np.float32)
    for i in range(len(sv_low)):
        sv_diag[i, i] = sv_low[i]
    
    new_v = uv_low @ sv_diag @ vv_low
    new_v = np.clip(new_v, 0, 255).astype(np.uint8)
    
    # 合并YUV通道
    new_yuv = cv2.merge([new_y, new_u, new_v])
    
    # 转回BGR空间
    new_image = cv2.cvtColor(new_yuv, cv2.COLOR_YUV2BGR)
    return new_image

def save_yuv_svd_data(decomposition_data, y_rank, uv_rank, output_dir):
    """保存YUV空间的SVD分解结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"yuv_svd_data_y{y_rank}_uv{uv_rank}.npz")
    
    # 使用压缩格式保存，并使用float16可进一步减小文件大小
    np.savez_compressed(filename, 
             uy=decomposition_data['Y']['U'].astype(np.float32), 
             sy=decomposition_data['Y']['S'].astype(np.float32), 
             vy=decomposition_data['Y']['V'].astype(np.float32),
             uu=decomposition_data['U']['U'].astype(np.float32), 
             su=decomposition_data['U']['S'].astype(np.float32), 
             vu=decomposition_data['U']['V'].astype(np.float32),
             uv=decomposition_data['V']['U'].astype(np.float32), 
             sv=decomposition_data['V']['S'].astype(np.float32), 
             vv=decomposition_data['V']['V'].astype(np.float32))
    
    return filename

def load_yuv_svd_data(filename):
    """加载YUV空间的SVD分解结果"""
    data = np.load(filename)
    decomposition_data = {
        'Y': {'U': data['uy'].astype(np.float32), 
              'S': data['sy'].astype(np.float32), 
              'V': data['vy'].astype(np.float32)},
        'U': {'U': data['uu'].astype(np.float32), 
              'S': data['su'].astype(np.float32), 
              'V': data['vu'].astype(np.float32)},
        'V': {'U': data['uv'].astype(np.float32), 
              'S': data['sv'].astype(np.float32), 
              'V': data['vv'].astype(np.float32)}
    }
    return decomposition_data

def compare_yuv_compression(image, output_dir):
    """比较不同Y/UV rank组合的效果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 测试不同的Y/UV rank组合
    y_ranks = [50, 100, 150]
    uv_ranks = [5, 10, 20]
    
    results = []
    
    for y_rank in y_ranks:
        for uv_rank in uv_ranks:
            # 处理图像
            decomposition_data = process_image_yuv(image, y_rank, uv_rank)
            
            # 保存SVD数据
            svd_filename = save_yuv_svd_data(decomposition_data, y_rank, uv_rank, output_dir)
            
            # 重建图像
            loaded_data = load_yuv_svd_data(svd_filename)
            reconstructed = reconstruct_from_yuv_svd(loaded_data)
            
            # 保存重建图像
            output_image = os.path.join(output_dir, f"yuv_recon_y{y_rank}_uv{uv_rank}.png")
            cv2.imwrite(output_image, reconstructed)
            
            # 计算指标
            from skimage.metrics import structural_similarity as ssim
            mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)
            psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
            
            # 转为灰度计算SSIM
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
            ssim_value = ssim(original_gray, reconstructed_gray)
            
            # 计算文件大小比
            original_size = os.path.getsize("fruits.png")
            compressed_size = os.path.getsize(svd_filename)
            size_ratio = compressed_size / original_size
            
            results.append({
                'y_rank': y_rank,
                'uv_rank': uv_rank,
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim_value,
                'size_ratio': size_ratio,
                'compressed_kb': compressed_size / 1024
            })
            
            print(f"YUV SVD (Y={y_rank}, UV={uv_rank}): MSE={mse:.2f}, PSNR={psnr:.2f}dB, SSIM={ssim_value:.4f}")
            print(f"  压缩比: {compressed_size/1024:.1f}KB / {original_size/1024:.1f}KB = {size_ratio:.2f}x")
    
    return results

def plot_yuv_metrics(results, output_dir):
    """绘制压缩指标图表，重点关注压缩比和MSE"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取不同的Y和UV rank值
    y_ranks = sorted(set(r['y_rank'] for r in results))
    uv_ranks = sorted(set(r['uv_rank'] for r in results))
    
    # 创建MSE和压缩比的热力图数据
    mse_data = np.zeros((len(y_ranks), len(uv_ranks)))
    size_ratio_data = np.zeros((len(y_ranks), len(uv_ranks)))
    
    # 填充数据
    for res in results:
        y_idx = y_ranks.index(res['y_rank'])
        uv_idx = uv_ranks.index(res['uv_rank'])
        mse_data[y_idx, uv_idx] = res['mse']
        size_ratio_data[y_idx, uv_idx] = res['size_ratio']
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # MSE热力图
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(mse_data, cmap='viridis')
    plt.title('MSE热力图 (Y vs UV rank)')
    plt.xlabel('UV Rank')
    plt.ylabel('Y Rank')
    plt.xticks(range(len(uv_ranks)), uv_ranks)
    plt.yticks(range(len(y_ranks)), y_ranks)
    plt.colorbar(im1, label='MSE')
    
    # 为每个单元格添加数值标签
    for i in range(len(y_ranks)):
        for j in range(len(uv_ranks)):
            plt.text(j, i, f'{mse_data[i, j]:.1f}', 
                     ha='center', va='center', color='w')
    
    # 压缩比热力图
    plt.subplot(2, 2, 2)
    im2 = plt.imshow(size_ratio_data, cmap='plasma')
    plt.title('压缩比热力图 (Y vs UV rank)')
    plt.xlabel('UV Rank')
    plt.ylabel('Y Rank')
    plt.xticks(range(len(uv_ranks)), uv_ranks)
    plt.yticks(range(len(y_ranks)), y_ranks)
    plt.colorbar(im2, label='压缩比')
    
    # 为每个单元格添加数值标签
    for i in range(len(y_ranks)):
        for j in range(len(uv_ranks)):
            plt.text(j, i, f'{size_ratio_data[i, j]:.2f}', 
                     ha='center', va='center', color='w')
    
    # MSE与Y-Rank关系 (不同UV-Rank)
    plt.subplot(2, 2, 3)
    for uv_idx, uv_rank in enumerate(uv_ranks):
        uv_data = [r for r in results if r['uv_rank'] == uv_rank]
        uv_data.sort(key=lambda x: x['y_rank'])
        
        y_rank_values = [r['y_rank'] for r in uv_data]
        mse_values = [r['mse'] for r in uv_data]
        
        plt.plot(y_rank_values, mse_values, 'o-', label=f'UV={uv_rank}')
    
    plt.title('MSE vs Y-Rank (不同UV-Rank)')
    plt.xlabel('Y Rank')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    
    # 压缩比与总Rank关系图
    plt.subplot(2, 2, 4)
    for r in results:
        r['total_rank'] = r['y_rank'] + 2 * r['uv_rank']  # 总秩 (Y+2*UV)
    
    # 按总秩排序
    sorted_results = sorted(results, key=lambda x: x['total_rank'])
    
    total_ranks = [r['total_rank'] for r in sorted_results]
    size_ratios = [r['size_ratio'] for r in sorted_results]
    mse_values = [r['mse'] for r in sorted_results]
    
    # 主坐标轴 - 压缩比
    ax1 = plt.gca()
    ax1.plot(total_ranks, size_ratios, 'b-o', label='压缩比')
    ax1.set_xlabel('总Rank (Y + 2*UV)')
    ax1.set_ylabel('压缩比', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 次坐标轴 - MSE
    ax2 = ax1.twinx()
    ax2.plot(total_ranks, mse_values, 'r-^', label='MSE')
    ax2.set_ylabel('MSE', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 添加标签，标记每个点的(Y,UV)值
    for i, r in enumerate(sorted_results):
        ax1.annotate(f'({r["y_rank"]},{r["uv_rank"]})', 
                    (total_ranks[i], size_ratios[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title('压缩比与MSE随总Rank变化')
    
    # 保存和显示图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yuv_compression_metrics.png'))
    plt.close()
    
    # 图像质量与文件大小的权衡关系
    plt.figure(figsize=(10, 6))
    compressed_sizes = [r['compressed_kb'] for r in results]
    
    # 绘制散点图，点大小与Y-rank成正比
    scatter = plt.scatter(compressed_sizes, mse_values, 
                         c=[r['uv_rank'] for r in results], 
                         s=[r['y_rank']*2 for r in results], 
                         cmap='viridis', 
                         alpha=0.7)
    
    # 添加标签
    for i, r in enumerate(results):
        plt.annotate(f'Y={r["y_rank"]},UV={r["uv_rank"]}', 
                   (compressed_sizes[i], mse_values[i]),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8)
    
    plt.colorbar(scatter, label='UV Rank')
    plt.title('图像质量与文件大小的权衡')
    plt.xlabel('压缩文件大小 (KB)')
    plt.ylabel('MSE (误差)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yuv_quality_size_tradeoff.png'))
    plt.close()

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 加载图像
    image = cv2.imread("fruits.png")
    if image is None:
        print("无法加载图像文件")
        return
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, 'yuv_results')
    
    # 比较YUV空间压缩效果
    results = compare_yuv_compression(image, output_dir)
    
    # 绘制指标图表
    plot_yuv_metrics(results, output_dir)
    
    # 选择一个效果较好的组合展示
    y_rank, uv_rank = 100, 10
    reconstructed = cv2.imread(os.path.join(output_dir, f"yuv_recon_y{y_rank}_uv{uv_rank}.png"))
    
    # 显示原图和重构图像
    cv2.imshow("Original", image)
    cv2.imshow(f"YUV SVD (Y={y_rank}, UV={uv_rank})", reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()