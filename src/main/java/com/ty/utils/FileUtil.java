package com.ty.utils;

import org.apache.commons.lang3.StringUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 文件工具类
 *
 * @Author Tommy
 * @Date 2021/12/7
 */
public class FileUtil {

    /**
     * 获取指定路径下符合条件的文件
     *
     * @param path 路径
     * @param suffix 后缀名
     * @return List<File>
     */
    public static List<File> getFiles(String path, String suffix) {

        List<File> dirList = new ArrayList<>();
        if (StringUtils.isNotBlank(path)) {
            final File directory = new File(path);
            if (directory.exists() && directory.isDirectory()) {
                FilenameFilter filter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.endsWith(suffix);
                    }
                };
                File[] files = directory.listFiles(filter);
                dirList.addAll(Arrays.asList(files));
            }
        }
        return dirList;
    }

    /**
     * 获取文件内容
     *
     * @param file
     * @return String
     */
    public static String getFileContent(File file) throws IOException {

        String content = StringUtils.EMPTY;
        if (null != file && file.exists()) {
            FileReader reader = new FileReader(file);
            BufferedReader br = new BufferedReader(reader);
            String line;
            while ((line = br.readLine()) != null)  {
                content += line;
            }
            br.close();
            reader.close();
        }
        return content;
    }
}
