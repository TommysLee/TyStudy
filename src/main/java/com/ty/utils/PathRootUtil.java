package com.ty.utils;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Path Root Util
 *
 * @Author Tommy
 * @Date 2024/5/5
 */
public class PathRootUtil {

    private static final String ROOT;

    static {
        ROOT = root();
    }

    public static String root() {
        String classesRoot = PathRootUtil.class.getClassLoader().getResource("").getPath();
        File dir = new File(classesRoot);
        String projectRoot = dir.getParentFile().getParentFile().getPath();
        return projectRoot + File.separator;
    }

    public static Path resolve(String relativePath) {
        return Paths.get(ROOT + relativePath);
    }
}
