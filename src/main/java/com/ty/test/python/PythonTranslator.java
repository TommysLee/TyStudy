package com.ty.test.python;

import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import lombok.extern.slf4j.Slf4j;

/**
 * Python Non-Standard Translator
 *
 * @Author Tommy
 * @Date 2024/5/2
 */
@Slf4j
public class PythonTranslator {

    private NDManager manager = NDManager.newBaseManager();

    public NDList processInput(Predictor<Input, Output> predictor) throws Exception {
        log.info("前置处理 Translator processInput 方法被调用......");
        Input input = new Input();
        input.addProperty("handler", "preprocess");
        predictor.predict(input);
        return null;
    }

    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        log.info("后置处理 Translator processOutput 方法被调用...... NDList = {}", list);

        Output output = new Output();
        return output;
    }
}
