package com.ty.test.python;

import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 基于 DJL Python Engine 执行Python Code 的非标Translator示例
 *
 * <p>
 *     Pyton Engine 不是一个真正的AI引擎，只是用来简化 “前置处理” 和 “后置处理”
 * </p>
 *
 * <p>
 *     因为当下Python在AI领域仍占主导地位，而AI推理最麻烦的部分也是“前后置处理”。
 *     各模型在Python下都有成熟的前后置处理逻辑，若要将其转DJL Java的成本高，而通过Python与DJL的结合，倒是个办法，这大概也是Python Engine存在的合理性。
 * </p>
 *
 * @Author Tommy
 * @Date 2024/5/2
 */
@Slf4j
public class PythonTest2 {

    static Gson gson = new Gson();

    static {
        Engine engine = Engine.getInstance();
        System.out.println(engine.getVersion() + "\t" + engine.getEngineName());
        System.out.println();

        Map<String, String> envs = System.getenv();
        System.out.println("系统环境变量 PYTHON_EXECUTABLE = " + envs.get("PYTHON_EXECUTABLE"));
        System.out.println();
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        PythonTranslator translator = new PythonTranslator();

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        // .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get(PythonTest2.class.getClassLoader().getResource("models/simple2").toURI()))
                        // .optModelUrls("djl://ai.djl.pytorch/resnet")
                        .optEngine("Python")
                        .build();


        ZooModel<Input, Output> model = criteria.loadModel();
        Predictor<Input, Output> predictor = model.newPredictor();

        translator.processInput(predictor);

        System.out.println("\r\n等待3秒后，多线程处理。。。。\r\n");

        Thread.currentThread().sleep(3000);

        // NDManager manager = NDManager.newBaseManager();
        NDManager manager = model.getNDManager();
        System.out.println(manager);

        // 多线程测试
        int nthreads = 3;
        ExecutorService executorService = Executors.newFixedThreadPool(nthreads);
        for (int i = 0; i < nthreads; i++) {
            executorService.execute(new Runnable() {
                @Override
                public void run() {
                    synchronized (PythonTest2.class) { // Python Engine只支持单线程，所以这要加锁
                        try {
                            Input input = new Input();
                            input.add("name", Thread.currentThread().getName());
                            Output output = predictor.predict(input);

                            BytesSupplier arrBytes = output.get("arr");
                            System.out.println(arrBytes);
                            System.out.println("JSON = " + output.getAsString("arr"));

                            List<Short> list = gson.fromJson(output.getAsString("arr"), new TypeToken<>(){});
                            System.out.println("List = " + list);
                            System.out.println("List Size: " + list.size());

                            int[] nums = list.stream().mapToInt(Number::shortValue).toArray();

                            NDArray ndArray = manager.create(nums, new Shape(nums.length));

                            log.info("推理输出Output = {}", output);
                            log.info("DJL Python 测试结束 {} {}", ndArray.getShape(), ndArray);
                        } catch (Exception e) {
                            log.error(e.getMessage(), e);
                        }
                    }
                }
            });
        }
    }
}
