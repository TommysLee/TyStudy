package com.ty.test;

import com.google.common.collect.Lists;
import com.ty.metrics.ClassificationReport;
import com.ty.regression.LogisticRegression;
import com.ty.transform.MinMaxScaler;
import com.ty.utils.Pandas;
import com.ty.utils.SKLearn;
import org.apache.commons.csv.CSVRecord;

import java.util.List;

/**
 * 逻辑回归模型实战（属于 机器学习 范畴）
 *
 * 相应的Python等价实现，请参考：src/python/逻辑回归模型实战.py
 *
 * @Author Tommy
 * @Date 2022/11/10
 */
public class LogisticRegressionTest {

    // 数据文件路径
    static final String path = "data/breast_cancer_data.csv";

    // 打印
    static void print(List<?> list) {
        list.forEach(System.out::println);
        System.out.println("记录数：" + list.size());
        System.out.println();
    }

    static void print(String name, Object value) {
        System.out.println(name + value);
        System.out.println();
    }

    // 测试
    public static void main(String[] args) throws Exception {

        // 读取CSV数据
        List<CSVRecord> dataset = Pandas.readCsv(path);
        //print(dataset);

        // 划分数据集和测试集（shuffle为true，即随机划分：80%数据集、20%测试集）
        List<CSVRecord> trainDataset = Lists.newArrayList();
        List<CSVRecord> testDataset = Lists.newArrayList();
        SKLearn.train_test_split(dataset, 0.2, true, trainDataset, testDataset);
        //print(trainDataset);

        // 提取特征数据X（所有的行 和 指定列【第0列至倒数第二列】）
        List<List<Double>> x_train = Pandas.iloc(trainDataset, 0, -1);  // 训练集特征数据
        List<List<Double>> x_test = Pandas.iloc(testDataset, 0, -1);    // 测试集特征数据
        // print(x_train);

        // 提取标签数据Y（根据列名提取）
        List<Double> y_train = Pandas.iloc(trainDataset, "target"); // 训练集标签数据
        List<Double> y_test = Pandas.iloc(testDataset, "target");   // 测试集标签数据
        // print(y_train);

        // 数据归一化处理
        MinMaxScaler.fit_transform(x_train); // 训练集归一化处理
        MinMaxScaler.fit_transform(x_test);  // 测试集归一化处理
        // print(x_train);

        // 逻辑回归模型搭建
        LogisticRegression lr = new LogisticRegression();
        lr.fit(x_train, y_train);

        // 打印模型的参数
        print("w = ", lr.getCoef_());
        print("b = ", lr.getIntercept_());
        //print("loss = ", lr.getLossesList());

        // 利用训练好的模型进行推理测试
        lr.setThresholds(0.3d); // 设置阈值
        List<Integer> pre_reslut = lr.predict(x_test);
        print("推理结果：", pre_reslut);

        // 打印预测结果的概率
        List<Double> pre_reslut_proba = lr.getPredict_proba();
        print("推理结果概率：", pre_reslut_proba);

        // 设置结果列表
        List<String> resultNames = Lists.newArrayList();
        for (Integer r : pre_reslut) {
            resultNames.add(1 == r? "恶性" : "良性");
        }

        // 打印阈值调整后的结果
        print("结果名称：", resultNames);

        // 输出结果的精确率、召回率、F1值
        ClassificationReport report = new ClassificationReport(y_test, pre_reslut, pre_reslut_proba);
        print("打印报告：\n", report);

        // 输出Loss值到文件
        lr.outputLosses();
    }
}
