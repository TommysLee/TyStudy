package com.ty.utils;

import java.awt.*;
import java.io.Serial;
import java.util.HashMap;
import java.util.Map;

/**
 * Color Util
 *
 * @Author Tommy
 * @Date 2024/5/10
 */
public class ColorUtil {

    private static Map<String, Color> LABEL_COLORS = new HashMap<>() {

        @Serial
        private static final long serialVersionUID = -6444008503314742586L;

        {
            // 中文
            put("人", new Color(63,81,181));
            put("背包", new Color(0,188,212));
            put("雨伞", new Color(74,20,140));
            put("手提包", new Color(0,96,100));
            put("领带", new Color(96,125,139));
            put("瓶子", new Color(230,81,0));
            put("酒杯", new Color(255,87,34));
            put("椅子", new Color(121,85,72));
            put("沙发", new Color(198,255,0));
            put("盆栽植物", new Color(76,175,80));
            put("餐桌", new Color(2,119,189));
            put("电视", new Color(213,0,249));
            put("笔记本电脑", new Color(94,53,177));
            put("鼠标", new Color(255,214,0));
            put("遥控器", new Color(29,233,182));
            put("键盘", new Color(249,168,37));
            put("手机", new Color(41,121,255));
            put("微波炉", new Color(57,73,171));
            put("烤箱", new Color(255,64,129));
            put("烤面包机", new Color(245,0,87));
            put("水槽", new Color(55,71,79));
            put("冰箱", new Color(120,144,156));
            put("书", new Color(124,77,255));

            // 英文
            put("person", new Color(63,81,181));
            put("backpack", new Color(0,188,212));
            put("umbrella", new Color(74,20,140));
            put("handbag", new Color(0,96,100));
            put("tie", new Color(96,125,139));
            put("bottle", new Color(230,81,0));
            put("wine glass", new Color(255,87,34));
            put("chair", new Color(121,85,72));
            put("couch", new Color(198,255,0));
            put("potted plant", new Color(76,175,80));
            put("dining table", new Color(2,119,189));
            put("tv", new Color(213,0,249));
            put("laptop", new Color(94,53,177));
            put("mouse", new Color(255,214,0));
            put("remote", new Color(29,233,182));
            put("keyboard", new Color(249,168,37));
            put("cell phone", new Color(41,121,255));
            put("microwave", new Color(57,73,171));
            put("oven", new Color(255,64,129));
            put("toaster", new Color(245,0,87));
            put("sink", new Color(55,71,79));
            put("refrigerator", new Color(120,144,156));
            put("book", new Color(124,77,255));
        }
    };

    public static final Color RED = new Color(233,30,99);

    /**
     * 根据标签获取颜色
     *
     * @param label 标签
     * @return Color
     */
    public static Color color(String label) {
        Color color = LABEL_COLORS.get(label);
        return null != color? color : RED;
    }
}
