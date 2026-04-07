package com.ty.test;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * AWS DJL 测试类
 *
 * <p>
 *     DJL是亚马逊在2019年宣布推出的开源的深度学习开发包
 * </p>
 *
 * @Author Tommy
 * @Date 2022/11/18
 */
@Slf4j
public class DJLTest {

    static NDManager manager = NDManager.newBaseManager();

    static void t1() {
        NDArray x = manager.arange(12);
        System.out.println(x);
        System.out.println("维度：" + x.getShape().dimension());
        System.out.println("维度长度：" + x.getShape().size());
        System.out.println();

        x = x.reshape(3, 4);
        System.out.println(x);
    }

    static void t2() {
        NDArray x = manager.zeros(new Shape(3, 4));
        System.out.println(x);
        System.out.println("维度：" + x.getShape().dimension());
        System.out.println("维度长度/元素个数：" + x.getShape().size());

        x = x.add(1).reshape(4, 3);
        System.out.println(x);
        System.out.println(x.get("-1"));
        System.out.println(x.hasGradient());
        System.out.println(x.sum());
    }

    static void t3() throws Exception {
        Table x_data = Table.read().csv(DJLTest.class.getClassLoader().getResource("data/breast_cancer_data.csv").getFile());
        Table y_data = x_data.selectColumns("target");
        x_data.removeColumns("target");

        System.out.println(x_data);
        System.out.println(y_data);

        // 转为NDArray格式
        NDArray x = manager.create(x_data.as().doubleMatrix());
        NDArray y = manager.create(y_data.as().doubleMatrix());

        System.out.println(x);
        System.out.println();
        System.out.println(y);

        Plot.show(LinePlot.create("图表", "次数", new double[]{1,2,3,4,5,6,7,8,9,10}, "Loss", new double[]{0,1,0,1,1,0,1,2,1,1, 0}), "chart");
    }

    public static void main(String[] args) throws Exception {
        t1();

        SequentialBlock block = new SequentialBlock();
    }
}
