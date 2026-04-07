package com.ty.utils;

import java.io.File;

/**
 * ONNX Util
 *
 * @Author Tommy
 * @Date 2024/5/31
 */
public class OnnxUtil {

    public static final String ONNXRUNTIME_NATIVE_PATH = "onnxruntime.native.path";

    /**
     * 设置ONNX引擎的路径/载入默认路径的ONNX引擎
     */
    public static void loadShared() {
        String path = System.getProperty("user.home") + File.separator + ".djl.ai/onnx";
        System.setProperty(ONNXRUNTIME_NATIVE_PATH, path);
    }
}
