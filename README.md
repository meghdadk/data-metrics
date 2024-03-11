<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

<h1 align="center">Data Metrics</h1>

## 📚 Description
This project aims to provide a set of metrics used to study deep learning models' behaviour w.r.t the datasets.

## 📊 The Metrics
- **Memorization**: As in [Feldman et al.](https://arxiv.org/abs/2008.03703)
- **C-score**: As in [Jiang et al.](https://arxiv.org/abs/2002.03206)
- **Learning-events**: As in [Toneva et al.](https://arxiv.org/pdf/1812.05159.pdf)
- **Loss curvature**: As in [Garg et al.](https://arxiv.org/pdf/2307.05831.pdf)
- **Activation-based**: INCOMPLETE

## 🚀 Usage
1. Make a copy of `configs/train.config` and set its parameters.
2. Run your favourite method as follows:

```bash
python [method.py] --config configs/config.yaml --exp-key <exp-id-in-your-config>
```

## 🤝 Contributing
Contributions are welcome! If you have any ideas or improvements, please open an issue or submit a pull request.

## 📝 License
This project is MIT licensed.