import os
import math
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# 添加项目根目录到Python路径
import sys
sys.path.append('.')

# 导入增强的谱图生成模块
try:
    from spectrum_generation import SpectraGenerator
    SPECTRUM_GENERATION_AVAILABLE = True
except ImportError:
    SPECTRUM_GENERATION_AVAILABLE = False
    print("Warning: spectrum_generation module not found. Using basic augmentation.")

# 默认参数设置
DEFAULT_MIN_ANGLE = 10.0   # 2θ最小值（度）
DEFAULT_MAX_ANGLE = 80.0   # 2θ最大值（度）
DEFAULT_STEP = 0.05        # 2θ步长（度），需与训练时一致
DEFAULT_WAVELENGTH = "CuKa"  # X射线波长，默认Cu Kα

def simulate_pattern_from_cif(cif_file, min_angle=DEFAULT_MIN_ANGLE, max_angle=DEFAULT_MAX_ANGLE,
                              step=DEFAULT_STEP, wavelength=DEFAULT_WAVELENGTH):
    """
    从单个CIF文件模拟XRD衍射图谱。返回2θ数组和对应强度数组（已归一化）。
    若未安装pymatgen或解析失败，则返回None。
    """
    # 读取晶体结构
    try:
        structure = Structure.from_file(cif_file)
    except Exception as e:
        print(f"[WARN] CIF解析失败: {cif_file}, 错误: {e}")
        return None
    # 使用pymatgen的XRDCalculator模拟衍射图
    xrd_calc = XRDCalculator(wavelength=wavelength)
    # 获取衍射峰列表（角度和强度）
    try:
        pattern = xrd_calc.get_pattern(structure, two_theta_range=(min_angle, max_angle))
    except Exception as e:
        print(f"[WARN] 模拟XRD失败: {cif_file}, 错误: {e}")
        return None
    two_thetas = np.array(pattern.x)
    intensities = np.array(pattern.y)
    if len(two_thetas) == 0:
        print(f"[WARN] {os.path.basename(cif_file)} 在范围 {min_angle}-{max_angle}° 内无衍射峰.")
        # 若无峰则返回全零数组
        num_points = math.floor((max_angle - min_angle) / step) + 1
        return np.linspace(min_angle, max_angle, num_points), np.zeros(num_points)
    # 将离散衍射峰转换为连续谱线：使用高斯展宽峰形
    num_points = math.floor((max_angle - min_angle) / step) + 1
    angles = np.linspace(min_angle, max_angle, num_points)
    profile = np.zeros(num_points)
    # 根据晶粒尺寸估计峰宽：较小晶粒 -> 峰更宽。简单采用Scherrer公式近似：FWHM (deg) ~ K * λ / (L * cosθ)
    # 这里取K=0.9, λ≈1.54Å(CuKa), L取较小值得到峰宽上限 ~0.3°, 大L峰宽下限 ~0.05°
    # 简化处理：统一选用一个中等晶粒尺寸(例如100 nm)计算基础峰宽，然后再由MIN_DOMAIN_SIZE/MAX_DOMAIN_SIZE调节
    base_domain = 100.0  # nm
    base_FWHM = 0.9 * 1.54 / (base_domain * 2)  # 估算在中等θ角度下的FWHM（简单近似）
    for theta, intensity in zip(two_thetas, intensities):
        # 计算当前峰对应的晶粒大小带来的FWHM
        # 将晶粒尺寸限制在[min,max]，随机取一个值以加入数据增强的多样性
        domain_size = base_domain
        # Scherrer公式简化：FWHM 与 1/domain 成正比
        fwhm = base_FWHM * (base_domain / domain_size)
        # 转换FWHM为标准差sigma（高斯）：FWHM = 2.355 * sigma
        sigma = fwhm / 2.355
        # 计算该衍射峰在angles网格上的强度分布（高斯形状）
        if sigma < 1e-6:
            # 如近似无展宽，则直接取最近的点
            idx = int(round((theta - min_angle) / step))
            if 0 <= idx < len(profile):
                profile[idx] += intensity
        else:
            # 生成高斯分布并累加到谱线上
            gauss = np.exp(-0.5 * ((angles - theta) / sigma) ** 2)
            # 使高斯峰的峰值等于 intensity
            gauss = intensity * gauss / gauss.max()
            profile += gauss
    # 归一化：将最高峰强度归一到1
    if profile.max() > 0:
        profile /= profile.max()
    return angles, profile

def load_dataset_from_cifs_enhanced(cif_dir, train_ratio=0.8, augment_each=50,
                                   min_angle=DEFAULT_MIN_ANGLE, max_angle=DEFAULT_MAX_ANGLE, step=DEFAULT_STEP):
    """
    从给定文件夹读取所有 CIF，使用增强的谱图生成模块生成数据集张量。
    - augment_each: 为每个物相生成多少条增强样本（>=1）。
    返回：(X_train, y_train), (X_val, y_val), 以及物相类别信息字典。
    """
    cif_files = [os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.lower().endswith('.cif')]
    cif_files.sort()
    if len(cif_files) == 0:
        raise FileNotFoundError(f"目录 {cif_dir} 中没有找到 CIF 文件。")
    
    # 计算预期的谱图长度
    expected_length = int((max_angle - min_angle) / step) + 1
    print(f"Expected spectrum length: {expected_length}")
    
    # 如果有增强模块，使用它来生成数据
    if SPECTRUM_GENERATION_AVAILABLE:
        print("使用增强的谱图生成模块...")
        generator = SpectraGenerator(
            reference_dir=cif_dir,
            num_spectra=augment_each,
            max_texture=0.6,
            min_domain_size=1.0,
            max_domain_size=100.0,
            max_strain=0.04,
            max_shift=0.25,
            impur_amt=70.0,
            min_angle=min_angle,
            max_angle=max_angle,
            separate=False  # 使用混合增强
        )
        
        # 生成增强的谱图
        all_spectra = generator.augmented_spectra
        
        # 列表存储所有样本光谱和标签
        patterns = []
        labels = []
        label_names = []  # 存储物相名称（如 "Formula_SG"）
        label_to_index = {}
        
        for i, (cif_file, spectra_group) in enumerate(zip(cif_files, all_spectra)):
            # 提取物相名：用 化学式_空间群 作为标识
            formula = os.path.splitext(os.path.basename(cif_file))[0]  # 文件名（可能含化学式）
            phase_name = formula
            # 若能获取空间群信息，则附加
            try:
                struct = Structure.from_file(cif_file)
                sg = struct.get_space_group_info()[0]  # 空间群符号
                # 去除空格和斜杠
                sg_label = sg.replace(" ", "").replace("/", "")
                phase_name = f"{struct.composition.reduced_formula}_{sg_label}"
            except Exception:
                pass
                
            if phase_name not in label_to_index:
                label_names.append(phase_name)
                label_to_index[phase_name] = len(label_names) - 1
            
            # 添加所有增强的谱图
            for spectrum in spectra_group:
                # 确保谱图形状正确
                if len(spectrum) > 0:
                    # 转换为一维数组
                    profile = np.array(spectrum).flatten()
                    # 调整长度以匹配预期长度
                    if len(profile) != expected_length:
                        # 如果长度不匹配，进行插值
                        old_x = np.linspace(min_angle, max_angle, len(profile))
                        new_x = np.linspace(min_angle, max_angle, expected_length)
                        profile = np.interp(new_x, old_x, profile)
                    # 归一化
                    if profile.max() > 0:
                        profile = profile / profile.max()
                    patterns.append(profile.astype(np.float32))
                    labels.append(label_to_index[phase_name])
    else:
        # 如果没有增强模块，回退到基本的数据加载和增强
        print("使用基本的数据加载和增强...")
        # from data_utils import load_dataset_from_cifs
        # return load_dataset_from_cifs(cif_dir, train_ratio, augment_each, min_angle, max_angle, step)
    
    patterns = np.array(patterns, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    # 打乱数据集顺序
    indices = np.arange(len(patterns))
    np.random.shuffle(indices)
    patterns = patterns[indices]
    labels = labels[indices]
    # 按 train_ratio 划分训练/验证集
    train_size = int(len(patterns) * train_ratio)
    X_train = patterns[:train_size]
    y_train = labels[:train_size]
    X_val = patterns[train_size:]
    y_val = labels[train_size:]
    # 将光谱形状调整为 (N, 1, L) 以匹配Conv1D输入
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]
    # 返回数据及类别信息
    class_info = {
        "class_names": label_names,
        "class_to_index": label_to_index,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "step": step
    }
    return (X_train, y_train), (X_val, y_val), class_info
