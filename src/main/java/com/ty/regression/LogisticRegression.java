package com.ty.regression;

import com.google.common.collect.Lists;
import com.ty.utils.NumberUtil;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.util.FastMath;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.ty.utils.NumberUtil.lamada;

/**
 * 逻辑回归模型（机器学习 的重要模型）
 *
 * 可以理解为：线性回归模型 + Sigmoid函数
 *
 * <p>
 *     理论说明，可参考：https://blog.51cto.com/greyfoss/5468674
 * </p>
 * <p>
 *     说明：引入正则化项的目的是防止模型过拟合。
 * </p>
 *
 * @Author Tommy
 * @Date 2022/11/10
 */
public class LogisticRegression {

    // 学习率
    private double r;

    // 最大训练轮次
    private int max_iter;

    // 特征的权重集合
    private List<Double> coef_ = Lists.newArrayList();

    // 偏置
    private double intercept_;

    // 每个轮次的损失值
    private List<Double> lossesList = Lists.newArrayList();

    // 特征数据
    private List<List<Double>> features;

    // 标签数据
    private List<Double> lables;

    // 当两次训练损失函数下降小于此值时提前结束训练
    private double tol = Math.pow(10, -6);
    private int tick = 10;

    // 权重初始值
    private double w;

    // 二分类阈值
    private double thresholds = 0.5;

    // 预测的概率
    private List<Double> predict_proba;

    // Sigmoid对象
    private Sigmoid sigmoid = new Sigmoid();

    // 是否多线程训练
    private boolean parallel;

    // 多线程数量
    private int threadCount = 4;

    // 多线程管理器
    private ExecutorService executorService;

    /**
     * 逻辑回归默认构造函数
     */
    public LogisticRegression() {
        this(0.1, 1, 0, 10000, true); //6120
    }

    /**
     * 逻辑回归构造函数
     *
     * @param r 学习率
     * @param w 初始权重
     * @param intercept_ 初始偏置
     * @param max_iter 最大训练轮次
     * @param parallel 使用多线程训练
     */
    public LogisticRegression(double r, double w, double intercept_, int max_iter, boolean parallel) {
        this.r = r;
        this.w = w;
        this.intercept_ = intercept_;
        this.max_iter = max_iter;
        this.parallel = parallel;

        if (parallel) {
            executorService = Executors.newFixedThreadPool(threadCount);
        }
    }

    /**
     * Sigmoid函数
     */
    double sigmoid(double z) {
        return sigmoid.value(z);
    }

    /**
     * 数学模型（用于前向传播）
     *
     * <p>
     *     公式：g(X) = 1 / (1 + e<sup>-z</sup>)
     *     <br/>
     *          z = W<sup>T</sup> * X + b
     *     <br/>
     *          Sigmoid函数习惯性写为：σ(W<sup>T</sup> * X + b)
     * </p>
     *
     * @param xdata 一个样本的特征数据
     * @return 返回函数模型值
     */
    double forward(List<Double> xdata) {
        double z = 0;
        for (int i = 0; i < xdata.size(); i++) {
            double w = coef_.get(i);
            z += w * xdata.get(i);
        }
        z += intercept_;
        return sigmoid(z);
    }

    /**
     * 逻辑回归的损失函数，又称为：交叉熵损失函数（带L2正则化项）
     * <br/>
     * 公式：J(w,b) = (-1 / m) * ∑(y<sub>i</sub> * logσ(W<sup>T</sup> * x<sub>i</sub> + b) + (1 - y<sub>i</sub>)log(1 - σ(W<sup>T</sup> * x<sub>i</sub> + b))) + (λ/2m) * ∑W<sup>2</sup>
     *
     * @return double
     */
    double loss() {
        int m = features.size();
        double sum = 0;
        for (int i = 0; i < m; i++) {
            List<Double> xdata = features.get(i);
            double y = lables.get(i);

            // 通过交叉熵损失函数，计算损失
            double z = forward(xdata);
            sum += (y * FastMath.log(z) + (1 - y) * FastMath.log(1 - z));
        }
        double loss = -1 * sum / m;

        // L2正则化
        double l2 = 0d;
        for (double wi : coef_) {
            l2 += wi * wi;
        }
        l2 = (lamada / (2 * m)) * l2;
        loss += l2;

        return NumberUtil.tofix8(loss);
    }

    /**
     * 利用梯度下降法，更新特征 x<sup>j</sup> 的参数 w<sup>j</sup>（带L2正则化项）
     *  <br/>
     * 公式：J'(w) = 1/m * ∑(σ(W<sup>T</sup> * x<sub>i</sub> + b) - y<sub>i</sub>) * x<sub>i</sub><sup>j</sup> + (λ/m) * w<sup>j</sup>
     * <br/>
     * 说明：x<sub>i</sub><sup>j</sup> 表示 第i行j列的特征值，也简记作 x<sup>j</sup>
     *
     * @param xjdata 一组特征数据
     * @param wj 特征的权重
     * @return 返回更新后的w
     */
    double gradient_wj(List<Double> xjdata, double wj) {
        int m = xjdata.size();
        double sum = 0;
        for (int i = 0; i < m; i++) {
            // 获取样本数据 和 对应的标签
            List<Double> xdata = features.get(i);
            double y = lables.get(i);

            // 套用公式计算w的偏导数
            sum += (forward(xdata) - y) * xjdata.get(i);
        }

        // 此特征的 w 偏导数 均值
        double jwj = sum / m + (lamada * wj / m);

        // 更新特征的w
        return NumberUtil.tofix8(wj - r * jwj);
    }

    /**
     * 利用梯度下降法，更新偏置b
     * <br/>
     * 公式：J'(b) = 1/m * ∑(σ(W<sup>T</sup> * x<sub>i</sub> + b) - y<sub>i</sub>)
     */
    void gradient_b() {
        int m = this.features.size();
        double sum = 0;
        for (int i = 0; i < m; i++) {
            // 获取样本数据 和 对应的标签
            List<Double> xdata = features.get(i);
            double y = lables.get(i);

            // 套用公式计算b的偏导数
            sum += (forward(xdata) - y);
        }

        // 此特征的 b 偏导数 均值
        double jb = sum / m;

        // 更新特征的b
        intercept_ = NumberUtil.tofix8(intercept_ - r * jb);
    }

    /**
     * 多线程模型训练
     */
    void multiFit(final List<List<Double>> colsData) {
        int colSize = colsData.size();

        // 计算：将特征数据分为 threadCount 份，每份应为多少
        int eachSize = (colSize - 1) / (threadCount - 1);

        // 一共为 threadCount 个组，计算：每组的起始索引
        List<int[]> eachGroup = new ArrayList<>(threadCount);
        int gcount = threadCount - 1;
        for (int g = 0; g < gcount; g++) {
            eachGroup.add(new int[] {g * eachSize, (g + 1) * eachSize});
        }
        eachGroup.add(new int[] {eachSize * gcount, colSize}); // 最后一组

        // 开启多线程训练
        CountDownLatch countDownLatch = new CountDownLatch(threadCount);
        eachGroup.forEach(group -> {
            executorService.submit(() -> {
                this.standaloneFit(colsData, group[0], group[1]);
                countDownLatch.countDown();
            });
        });
        try {
            countDownLatch.await(); // 等待所有线程执行结束
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 单线程模型训练
     */
    void standaloneFit(List<List<Double>> colsData, int start, int end) {
        for (int j = start; j < end; j++) {
            List<Double> xjdata = colsData.get(j); // 第j列特征数据

            // 更新第j列参数w
            double wj = coef_.get(j);
            wj = this.gradient_wj(xjdata, wj);
            coef_.set(j, wj);
        }
    }

    /**
     * 训练结束
     */
    void fitOver() {
        if (parallel) {
            executorService.shutdown();
        }
    }

    /**
     * 模型训练
     *
     * @param features  训练的特征数据
     * @param lables    对应的标签数据
     */
    public void fit(List<List<Double>> features, List<Double> lables) {
        this.features = features;
        this.lables = lables;

        // 列数(一列即一个特征)
        int colSize = features.get(0).size();

        // 初始化一组模型参数w
        for(int j = 0; j < colSize; j++) {
            coef_.add(this.w);
        }

        // 提取各列特征数据
        List<List<Double>> colsData = Lists.newArrayList();
        for (int j = 0; j < colSize; j++) {
            List<Double> xjdata = Lists.newArrayList(); // 第j列特征数据
            for (int i = 0; i < features.size(); i++) {
                xjdata.add(features.get(i).get(j));
            }
            colsData.add(xjdata);
        }

        // 训练
        long begin = System.currentTimeMillis(), end = 0;
        for(int loop = 0; loop < max_iter; loop++) {
            // 计算损失
            double loss = this.loss();
            lossesList.add(loss);
            if (loop > 0) {
                double prevLoss = lossesList.get(lossesList.size() - 2); // 上一次的损失值
                if (Math.abs(prevLoss - loss) <= tol) {
                    tick--;
                }

                if (tick == 0) {
                    end = System.currentTimeMillis();
                    System.out.println("训练结束，轮次：" + (loop + 1) + "\t耗时：" + NumberUtil.minusAndMs2S(begin, end) + " 秒.\n");
                    fitOver();
                    return;
                }
            }

            // 更新各列的权重wj
            if (parallel) {
                this.multiFit(colsData);
            } else {
                this.standaloneFit(colsData, 0, colSize);
            }

            // 更新参数b
            this.gradient_b();
        }
        end = System.currentTimeMillis();
        System.out.println("超出最大训练轮次：" + max_iter + ", 停止训练.耗时：" + NumberUtil.minusAndMs2S(begin, end) + " 秒.\n");
        fitOver();
    }

    /**
     * 模型推理
     *
     * @param features 特征数据
     * @return List<Integer> 返回推理结果
     */
    public List<Integer> predict(List<List<Double>> features) {
        predict_proba = Lists.newArrayList();
        features.forEach(item -> {
            predict_proba.add(this.forward(item));
        });

        List<Integer> pre_reslut = Lists.newArrayList();
        predict_proba.forEach(proba -> {
            pre_reslut.add(proba > thresholds? 1 : 0);
        });
        return pre_reslut;
    }

    /**
     * 获取特征的权重集合
     *
     * @return List<Double>
     */
    public List<Double> getCoef_() {
        return this.coef_;
    }

    /**
     * 获取模型的偏置b
     *
     * @return double
     */
    public double getIntercept_() {
        return this.intercept_;
    }

    /**
     * 获取每轮训练的Loss值
     *
     * @return List<Double>
     */
    public List<Double> getLossesList() {
        return lossesList;
    }

    /**
     * 获取二分类阈值
     *
     * @return thresholds
     */
    public double getThresholds() {
        return thresholds;
    }

    /**
     * 设置二分类阈值
     *
     * @param thresholds 二分类阈值
     */
    public void setThresholds(double thresholds) {
        this.thresholds = thresholds;
    }

    /**
     * 获取预测的概率
     *
     * @return List<Double>
     */
    public List<Double> getPredict_proba() {
        return predict_proba;
    }

    /**
     * 输出Loss值到文件
     */
    public void outputLosses() {
        String path = this.getClass().getClassLoader().getResource("").getFile() + "losses.txt";
        StringBuilder tickBuilder = new StringBuilder("tick=[");
        StringBuilder lossBuilder = new StringBuilder("loss=[");
        for (int i = 0; i < this.lossesList.size(); i++) {
            tickBuilder.append(i+1).append(",");
            lossBuilder.append(lossesList.get(i)).append(",");
        }
        tickBuilder.delete(tickBuilder.length() - 1, tickBuilder.length()).append("]");
        lossBuilder.delete(lossBuilder.length() - 1, lossBuilder.length()).append("]");

        // 写入文件
        try (FileWriter fw = new FileWriter(path); BufferedWriter bw = new BufferedWriter(fw)) {
            bw.write(tickBuilder.toString());
            bw.newLine();
            bw.newLine();
            bw.write(lossBuilder.toString());
            bw.newLine();
            bw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
