package com.ty.test.python;

import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.nio.file.Paths;
import java.util.Map;

/**
 * 基于 DJL Python Engine 执行Python Code 的简单示例
 *
 * @Author Tommy
 * @Date 2024/4/29
 */
public class PythonTest {

    static {
        setDefaultEngine();
    }

    /**
     * 设置默认的引擎
     */
    static void setDefaultEngine() {
        System.setProperty("ai.djl.default_engine", "Python");
        // System.setProperty("ai.djl.default_engine", PpEngine.ENGINE_NAME);
        // System.setProperty("ai.djl.default_engine", PtEngine.ENGINE_NAME);
    }

    /**
     * 打印引擎信息
     */
    static void printEngineInfo() {
        Engine engine = Engine.getInstance();
        System.out.println(engine.getVersion());
        System.out.println(engine.getEngineName());
    }

    /**
     * 打印系统环境变量
     */
    static void printEnVar() {
        Map<String, String> envs = System.getenv();
        // for (Map.Entry<String, String> entry : envs.entrySet()) {
        //     System.out.println(entry.getKey() + " = " + entry.getValue());
        // }
        System.out.println(envs.get("PYTHON_EXECUTABLE"));
    }

    /**
     * 执行Python Code
     */
    static void t1() throws Exception {
        ZooModel<Input, Output> model;
        Predictor<Input, Output> predictor;

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get(PythonTest.class.getClassLoader().getResource("models/simple").toURI()))
                        .optEngine("Python")
                        .build();
        model = criteria.loadModel();
        predictor = model.newPredictor();

        System.out.println("模型初始化完毕");

        Input input = new Input();
        input.add("name", "Java17");
        input.addProperty("home", "MyHome China");

        Output output = predictor.predict(input);

        System.out.println("推理结束");

        if (output.getCode() != 200) {
            throw new TranslateException("Python preprocess() failed: " + output.getMessage());
        }

        System.out.println(output.getMessage());
        System.out.println(output.getProperty("message", "Empty"));
        System.out.println("Result = " + output.getAsString(0));
        System.out.println(output.getAsString("arr"));

        System.out.println("推理成功");

        input = new Input();
        input.add("name", "Nice！over");
        output = predictor.predict(input);
        System.out.println("Result2 = " + output.getAsString(0));
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        printEngineInfo();
        printEnVar();

        t1();
    }
}
