# 🎨 傅里叶级数绘图工具 (Fourier Series Drawing)

![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📝 项目简介

这个项目利用**傅里叶级数**（Fourier Series）将任意 SVG 图像转换为由旋转向量绘制的动画。通过数学的魔力，我们可以用简谐运动的叠加来绘制复杂的图形，展示数学之美！✨

本项目默认使用**重庆大学（CQU）的中文校徽**作为示例，展示了傅里叶变换将静态图像转换为动态绘制过程的能力。

![CQU Logo 演示](./demo/cqu_logo_demo.gif)

## ✨ 特性

- 🖼️ **SVG 转换**：将任意 SVG 图像转换为傅里叶级数表示
- 🚀 **多进程计算**：利用并行计算加速系数计算过程
- 💾 **智能缓存**：自动缓存计算结果，避免重复计算
- 🎮 **实时可视化**：使用 Pygame 实现动态绘制过程
- 🔄 **平滑插值**：对点进行预处理，使轨迹更加平滑
- 🌈 **自定义参数**：灵活调整向量数量、积分精度等参数

## 🛠️ 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 🚀 使用方法

1. 将你的 SVG 文件放入 `svg` 目录（默认使用重庆大学中文校徽）
2. 运行主程序：

```bash
python Fourier.py
```

3. 观看动态生成的傅里叶变换图像

## 🎮 操作说明

- **ESC 键**：退出程序
- **鼠标右键**：退出程序
- 观察绿色轨迹显示傅里叶变换的绘制过程

## 🔧 参数说明

可以在 `Fourier.py` 的 `__main__` 部分调整以下参数：

```python
# 进一步增加点的数量以获得更精细的轮廓
svg = Svg2points('CQU', 500, 1, show=False)

# 大幅增加向量数量和积分精度以提高还原度，启用缓存和并行计算
coeff = CalCoeff(svg, 800, vec_num=500, int_num=2000, use_cache=True, workers=None)

# 进一步放慢周期时间以便更清晰地观察
Visualization(20, coefficient=coeff, times=2)
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `point_num` | SVG 路径上提取的点数量 | 300-500 |
| `vec_num` | 傅里叶级数的向量数量 | 300-1000 |
| `int_num` | 积分计算的精度 | 1000-5000 |
| `use_cache` | 是否使用缓存 | True |
| `workers` | 并行计算的进程数 | None (自动) |

## 📊 工作原理

1. **SVG 解析**：从 SVG 文件中提取路径点
2. **点预处理**：对点进行平滑和插值处理
3. **傅里叶变换**：计算傅里叶级数系数
4. **向量可视化**：使用旋转向量绘制图形

傅里叶级数可以将任何周期函数表示为简谐函数的无穷和：

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nt) + b_n \sin(nt) \right]$$

## 📁 项目结构

```
项目目录/
├── Fourier.py       # 主程序文件
├── svg/             # SVG 文件目录
├── points/          # 提取的点数据
├── json/            # 傅里叶系数 JSON 文件
├── cache/           # 计算缓存
└── README.md        # 项目说明
```

## 💻 系统要求

- Python 3.6+
- 支持 pygame 的操作系统（Windows/macOS/Linux）

## 🙏 致谢

- 感谢 [3Blue1Brown](https://www.3blue1brown.com/) 的傅里叶级数可视化视频的启发
- 感谢所有开源库的贡献者

---

⭐ 如果这个项目对你有帮助，请给它一个星标！⭐
