package com.ty.utils;

import com.google.common.collect.Lists;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.Reader;
import java.util.List;

/**
 * 文件读取工具类
 *
 * @Author Tommy
 * @Date 2022/11/9
 */
public class Pandas {

    /**
     * 读取标准CSV文件
     *
     * @param path 文件相对路径
     * @return List<CSVRecord>
     */
    public static List<CSVRecord> readCsv(String path) throws Exception {
        Reader csvReader = new FileReader(Pandas.class.getClassLoader().getResource(path).getFile());
        CSVParser csvParser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(csvReader);
        return csvParser.getRecords();
    }

    /**
     * 从CSV数据集提取数据
     *
     * @param dataset 数据集
     * @param start 列起始位置
     * @param end   列结束位置
     * @return List<List<Double>>
     */
    public static List<List<Double>> iloc(List<CSVRecord> dataset, int start, int end) {
        List<List<Double>> values = Lists.newArrayList();
        int size = dataset.get(0).size();
        int cend = end >= 0? end : (size + end);

        dataset.stream().forEach(item -> {
            List<Double> itemV = Lists.newArrayList();
            for (int i = start; i < cend; i++) {
                itemV.add(Double.valueOf(item.get(i)));
            }
            values.add(itemV);
        });
        return values;
    }

    /**
     * 从CSV数据集提取数据
     *
     * @param dataset 数据集
     * @param headerName 列名
     * @return List<Double>
     */
    public static List<Double> iloc(List<CSVRecord> dataset, String headerName) {
        List<Double> colValues = Lists.newArrayList();
        dataset.stream().forEach(item -> {
            colValues.add(Double.valueOf(item.get(headerName)));
        });
        return colValues;
    }
}
