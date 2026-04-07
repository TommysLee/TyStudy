package com.ty.utils;

import lombok.extern.slf4j.Slf4j;

/**
 * 线程工具
 *
 * @Author Tommy
 * @Date 2024/5/10
 */
@Slf4j
public class ThreadUtil {

    /**
     * 线程休眠
     */
    public static void sleep(int ms) {
        try {
            Thread.sleep(ms);
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }
    }
}
