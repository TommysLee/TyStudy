package com.ty.test.numpy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * NDArray 测试
 *
 * @Author Tommy
 * @Date 2024/5/2
 */
public class NDArrayTest {

    static NDManager manager = NDManager.newBaseManager();

    /**
     * 创建NDArray
     */
    static void t1() {
        NDArray array = manager.create(new int[] {1,2,3,4,5});
        System.out.println(array);
        System.out.println(array.getShape());
        System.out.println(array.getShape().size());
        System.out.println(array.getShape().dimension());
    }

    /**
     * 创建NDArray（多维）
     */
    static void t2() {
        int[][] arr = new int[5][6];
        int tick = 1;
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                arr[i][j] = tick++;
            }
        }

        NDArray array = manager.create(arr);
        array = array.reshape(5, 2, 3);

        System.out.println(array);
        System.out.println(array.getShape());
        System.out.println(array.getShape().size());
        System.out.println(array.getShape().dimension());

        for (int i = 0; i < array.getShape().dimension(); i++) {
            System.out.println("维度 " + i + " 大小 " + array.getShape().size(i));
        }

        NDArray subArray = array.get(manager, new NDIndex("0, 1"));
        System.out.println(subArray);

        Number[] numbers = subArray.toArray();
        for (int i = 0; i < numbers.length; i++) {

        }
        int[] intArr = Arrays.stream(numbers).mapToInt(Number::intValue).toArray();
        Arrays.stream(intArr).forEach(System.out::println);
    }

    /**
     * 创建NDArray From JSON
     */
    static void t3() {
        String jsonText = "[" +
                "[" +
                    "[ 1,  2,  3],\n" +
                    "[ 4,  5,  6]\n" +
                "]" +
            "]";

        Gson gson = new Gson();
        List<List<List<Integer>>> list = gson.fromJson(jsonText, new TypeToken<>() {});
        System.out.println(list);
        System.out.println();

        int[] nums = new int[list.size() * list.get(0).size() * list.get(0).get(0).size()];
        AtomicInteger index = new AtomicInteger();
        list.forEach(subList -> {
            subList.forEach(ssubList -> {
                ssubList.forEach(n -> {
                    nums[index.getAndIncrement()] = n;
                });
            });
        });

        NDArray ndArray = manager.create(nums, new Shape(list.size(), list.get(0).size(), list.get(0).get(0).size()));
        System.out.println(ndArray);
    }

    static void t4() {
        NDArray ndArray = manager.arange(1, 10).reshape(3, 3);
        System.out.println(ndArray);
        System.out.println();

        NDArray oneNdArray = manager.ones(new Shape(5, 2), DataType.INT32);
        System.out.println(oneNdArray);
        System.out.println();

        NDArray eightNdArray = manager.full(new Shape(5, 2), 8);
        System.out.println(eightNdArray);
        System.out.println();

        // INT32 ---> INT8
        NDArray randomNdArray = manager.randomInteger(10, 50, new Shape(5, 2), DataType.INT32);
        randomNdArray = randomNdArray.toType(DataType.INT8, false);
        System.out.println(randomNdArray);
    }

    /*
     * 测试
     */
    public static void main(String[] args) {
        t4();

        manager.close();
    }
}
