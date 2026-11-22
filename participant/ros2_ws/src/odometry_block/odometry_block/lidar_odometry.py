import numpy as np

def convert_ranges_to_coords(
    ranges: list[float],  
    angle_min: float, 
    angle_max: float,
    angle_increment: float
):
    """
    Функция, переводящая массив из пространства полярных координат в 
    пространство декартовых координат [(r, phi) -> (x, y)].
    :param ranges: list[float] - множество отмеренных ЛиДаром расстояний, длиной N
    :param angle_min: float - минимальный угол
    :param angle_max: float - максимальный угол
    :param angle_increment: float - шаг угла
    :return: np.ndarray - матрица координат размерностью [N, 2]
    """
    pass


def estimate_delta(
    prev_points: np.ndarray, 
    curr_points: np.ndarray
):
    """
    Функция, вычисляющая изменения координат на основе двух массивов измерений ЛиДАРа
    :param prev_points: np.ndarray - предыдущее измерение ЛиДАРа
    :param curr_points: np.ndarray - нынешнее измерение ЛиДАРа
    :return: np.ndarray - посчитанные изменения координат робота
    """
    pass
