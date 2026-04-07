package com.ty.utils;

import org.apache.commons.csv.CSVRecord;

import java.util.Collections;
import java.util.List;

/**
 * 模拟python sklearn
 *
 * @Author Tommy
 * @Date 2022/11/9
 */
public class SKLearn {

    /**
     * 随机划分训练集和测试集
     *
     * @param dataset   待划分的数据集
     * @param test_size 测试集比例
     * @param trainDataset  划分后的训练集
     * @param testDataset   划分后的测试集
     */
    public static void train_test_split(List<CSVRecord> dataset, double test_size, List<CSVRecord> trainDataset, List<CSVRecord> testDataset) {
        train_test_split(dataset, test_size, true, trainDataset, testDataset);
    }

    /**
     * 划分训练集和测试集
     *
     * @param dataset   待划分的数据集
     * @param test_size 测试集比例
     * @param shuffle   是否随机划分
     * @param trainDataset  划分后的训练集
     * @param testDataset   划分后的测试集
     */
    public static void train_test_split(List<CSVRecord> dataset, double test_size, boolean shuffle, List<CSVRecord> trainDataset, List<CSVRecord> testDataset) {
        // 随机排序
        if (shuffle) {
            Collections.shuffle(dataset);
        }
        int size = dataset.size();

        // 测试集个数
        int testCount = Math.round(Double.valueOf(size * test_size).floatValue());

        // 生成训练集
        trainDataset.addAll(dataset.subList(0, size - testCount));

        // 生成测试集
        testDataset.addAll(dataset.subList(size - testCount, size));
    }
}
