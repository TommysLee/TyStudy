package com.ty.test;

/**
 * 线性回归模型实战（属于 机器学习 范畴）
 *
 * 相应的Python等价实现，请参考：src/python/线性回归模型实战.py
 *
 * @Author Tommy
 * @Date 2022/11/4
 */
public class LinearRegressionTest {

    // 特征数据
    static double[] features = new double[] {1, 2, 3};

    // 标签(结果)
    static double[] labels = new double[] {2, 4, 6};

    // W参数初始化
    static double w = 4d;

    // 学习率
    static double a = 0.01;

    /**
     * 模型（前向传播）
     */
    static double forward(double x) {
        return w * x;
    }

    /**
     * 损失函数（均值平方法）
     */
    static double loss() {
        // 根据当前w计算损失（均值）
        double totalW = 0d;
        for (int i = 0; i < features.length; i++) {
            totalW += Math.pow(forward(features[i]) - labels[i], 2); // 预测值- 实际值，然后平方
        }
        return totalW / features.length;
    }

    /**
     * 利用梯度下降法更新W参数
     */
    static double gradient() {
        double ws = 0d;
        for (int i = 0; i < features.length; i++) {
            ws += w - 2 * a * features[i] * (forward(features[i]) - labels[i]);
        }
        w = ws / features.length;
        return w;
    }

    /**
     * 训练
     */
    static void trainning(int loop) {
        for (int i = 1; i <= loop; i++) {
            // 更新参数
            double wi = gradient();

            // 计算损失
            double loss = loss();

            // 输出日志
            System.out.println("第 " + i + " 轮：w=" + w + ", loss=" + loss);
        }
        System.out.println("\r\n训练结束，w=" + w);
    }

    /**
     * 预测
     */
    static double predict(double x) {
        double result = forward(x);
        return result;
    }

    /*
     * 测试运行
     */
    public static void main(String[] args) {
        // 训练100轮
        trainning(100);

        // 预测
        double x = 4;
        double y = predict(x);
        System.out.println("\r\n预测: x=" + x + ", y=" + y + " (近似值: " + Math.round(y) + ")");

    }
}
