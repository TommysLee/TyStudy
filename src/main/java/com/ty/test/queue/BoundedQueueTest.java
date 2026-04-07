package com.ty.test.queue;

import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * 有界队列测试
 *
 * @Author Tommy
 * @Date 2024/5/9
 */
@Slf4j
public class BoundedQueueTest {

    // 有界阻塞队列对象
    static final LinkedBlockingQueue<Integer> QUEUE = new LinkedBlockingQueue<>(1);

    static void startThread() {
        new Thread(() -> {
            try {
                while (true) {
                    log.info("\t\t\t\t\t\t应答：{}", QUEUE.take());
                    sleep(1000);
                }
            } catch (Exception e) {
                log.error(e.getMessage(), e);
            }
        }).start();
    }

    static void send(int n) throws Exception {
        boolean flag = QUEUE.offer(n);
        log.info("发送：{}\t{}", n, flag);
        sleep(200);
    }

    static void sleep(int ms) throws Exception {
        Thread.currentThread().sleep(ms);
    }

    public static void main(String[] args) throws Exception {
        startThread();

        int tick = 0;
        do {
            try {
                send(++tick);
            } catch (Exception e) {
                log.error(e.getMessage(), e);
            }
        } while (tick < 100);

        sleep(1500);
        System.out.println("有界阻塞队列·测试结束");
        System.exit(0);
    }
}
