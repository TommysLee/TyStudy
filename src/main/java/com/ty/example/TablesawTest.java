package com.ty.example;

import org.apache.commons.lang3.RandomUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

/**
 * TableSaw类似Python Pandas
 *
 * @Author Tommy
 * @Date 2022/12/4
 */
public class TablesawTest {

    /**
     * 创建Table对象
     */
    static Table createTable() {
        Table table = Table.create("能耗数据表", DoubleColumn.create("Epi"), DoubleColumn.create("P"));
        for (int r = 0; r < 50; r++) {
            Row row = table.appendRow();
            for (int i = 0; i < table.columnCount(); i++) {
                row.setDouble(i, RandomUtils.nextDouble(100, 500));
            }
        }
        return table;
    }

    /**
     * 打印表格
     */
    static void printTable(Table table) {
        System.out.println(table.structure().printAll());
        System.out.println();
        System.out.println(table.print());
    }

    /**
     * 导出CSV
     */
    static void exportCsv(Table table, String path) {
        table.write().csv(path);
    }

    /*
     * 测试
     */
    public static void main(String[] args) {
        Table table = createTable();
        printTable(table);
        exportCsv(table, "D:/epi_data.csv");
    }
}
