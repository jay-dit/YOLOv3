# YOLOv3
<div style='padding-top:2em;'>Результат!</div>
<img src="img\1.png">
<div style='padding-top:1em;'>А теперь немного о том как я к нему пришел</div>

<h3 style='color: black; font-weight: bold; font-family: Arial;'>
        <center>Реализация</center>
</h3>
<div>Весь код содержится в source/model</div>
<ul>
  <li style='padding-top:1em;'>Реализован Darknet_BN_Leaky_Relu слой</li>
  <li style='padding-top:1em;'>Реализован Residual слой</li>
  <li style='padding-top:1em;'>Реализовано объединение нескольких Residual слоёв вместе</li>
  <li style='padding-top:1em;'>Реализована модель YoloV3 tiny</li>
  <img width="500" height="900" src="img\model.jpg">
  <div style='padding-top:1em;font-size:90%;'>На данной картинке был произведен Reshape выходных слоев для их упрощенного вывода, однако обучение проводилось на модели с нормальными выходами</div>
  <li style='padding-top:2em;'>Так же был реализована кастомная функция ошибок</li>
  <img width="500" height="300" src="img\loss.png">
  <div style='padding-top:1em;font-size:90%;'>Взято из работы You Only Look Once:Unified, Real-Time Object Detection</div>
</ul>


<h3 style='color: black; font-weight: bold; font-family: Arial; padding-top:3em;'>
        <center>Датасет</center>
</h3>
<div style='padding-top:1em;'>После того, как модель была построена, я сгенерировал обучающий датасет</div>
<div style='padding-top:1em;'>Экзогенная переменная - одна картинка содержит рандомный фон и N количество пончиков на ней</div>
<div style='padding-top:1em;'>Эндогенная переменная - матрица 13x13x6 / 26x26x6 (Anchor boxes)</div>
<div style='padding-top:1em;'>Полный процесс генерации описан в source/data_generation</div>

<h3 style='color: black; font-weight: bold; font-family: Arial; padding-top:3em;'>
        <center>Обучение</center>
</h3>
<div style='padding-top:1em;'>Обучение производиось с использование адаптивного шага обучения</div>
<div style='padding-top:1em;'>Всего 100 эпох (около часа обучения на Видеокарте)</div>
<div style='padding-top:1em;'>После чего был получен вот такой результат:</div>
<img src="img\1.png">
<img src="img\2.png">
<div style='padding-top:1em;'>Конечно никакой практической применимость от этого нет</div>
<div style='padding-top:1em;'>Однако этот результат показывает, что код полностью рабочий и может использоваться на любых данных</div>
<div style='font-size:70%;'>______________________________</div>
<div style='font-size:70%;'>Lagutov Vladimir 2022</div>
