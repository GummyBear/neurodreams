var data, labels;
var t, layer_defs, net, trainer;
var ori_canvas, nn_canvas, ori_ctx, nn_ctx, oridata;
var arrf;
var arr_x = 0;

var batches_per_iteration = 100;
var mod_skip_draw = 100;
var smooth_loss = -1;
var intervalID;

time_start = 0;// Начало просчета картинки
time_finish = 0;// Конец просчета картинки
time_per_pic = 0;// Время просчета одной картинки
time_current_FPS = 0;// Текущий FPS

var counter = 0;// Номер текущей итерации (1 из 500, например).
var files_current = 1;// Номер текущей картинки (начинаем с первой)
var files_total = 1;// Всего выбрали картинок (сколько их в папке) 
net_param_temp = "";// Кусок строки параметров сети (только число слоев и нейронов)



/////////////////////////////////////////Net config//////////////////////////////////////////

flag_GPU = 0;// Найден или нет GPU
flag_debug = 0;// Показывать или нет отладочную информацию
time_max_FPS = 0.2;// Если FPS больше (FPS = 0.1 -> одна картинка за десять секунд), то можно увеличить сложность сети

flag_heritage = 1;// Наследуем предыдущую картинку (1) или нет (0) (пока не реализовано)
flag_rotate = 1;// Поворачиваем исходую картинку (1) или нет (0)

step   = 0.01;// Original = 0.01 (шаг обучения сети)
mom    = 0.9; // Original = 0.9 (момент)
batch  = 5; //   Original = 5 (???)

L_min = 1;// Минимальное число слоев
L_current = L_min;// Текущее число слоев
L_max = 15;// Максимальное число слоев (16 - максимум?)

N_min = 80;// Минимальное число нейронов в слое
N_current = N_min;// Текущее число нейронов в слое
N_max = 80;// Максимальное число нейронов в слое

total_neurons = 1;// Всего нейронов в сети (на старте один нейрон)
add_neurons = 15;// Сколько нейронов добавляем в сеть при ее усложнении (15?)

var sz_x = 500;// Выходное разрешение X
var sz_y = 500;// Выходное разрешение Y

var sz_g = Math.sqrt(sz_x*sz_x + sz_y*sz_y);// Размер гипотенузы (нужно для поворота картинки)
var sz_m = sz_g/sz_x;// Масштаб (нужно для поворота картинки)

var iteration = 500;// Шаг сохранения картинок (500 по умолчанию, сохраняем каждую пятую картинку). 

/////////////////////////////////////////Net config//////////////////////////////////////////



function create_neural_net(neurons, net_type) {// Функция создающая сеть одного из четырех типов

net_param_temp = "";
total_neurons = 0;


if (net_type == 0){// Создаем фиксированную тут сеть (без ограничений, можно забить любые параметры)

total_neurons = 666;
L_current = 666;
net_param_temp = "666";

t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2}); // 2 inputs: x, y \n\
layer_defs.push({type:'fc', num_neurons:100, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:90, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:80, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:70, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:60, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:50, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:40, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:30, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:10, activation:'relu'});\n\
layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b \n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:step, momentum:mom, batch_size:batch, l2_decay:0.0});\n\
";

}


if (net_type == 1){// Создаем плоскую сеть (типа [3, 3, 3, 3, 3]), с учетом максимального числа neurons
t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});" + "\n"// 2 inputs: x, y
for (x = 1; x<=L_current; x++) {// Число слоев
	N = Math.round(neurons/L_current);// Число нейронов в слое
	if (N>N_max){// Отсечка, если нейронов больше максимума
		N=N_max;
	}
	L_current = Math.ceil(neurons/N);//Округляем текущее число слоев в большую сторону!
	if (L_current>L_max){// Отсечка, если слоев больше максимума
		L_current=L_max;
	}
	//alert("x=" + x + " neurons= " + neurons + " N=" + N + " L_current=" + L_current);
	if ((total_neurons + N)>neurons){// Если перебрали с количеством нейронов, то чуть именьшаем их число
		N = neurons - total_neurons;	
	}
	total_neurons = total_neurons + N;
	if (N!=0){// Добавляем нейрон в сеть, только если он не нулевой
		net_param_temp = net_param_temp + N + ", ";
	}
  	t = t + "layer_defs.push({type:'fc', num_neurons:" + N + ", activation:'relu'});" + "\n";
}
t = t + "layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b \n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:step, momentum:mom, batch_size:batch, l2_decay:0.0});\n\
";
}




if (net_type == 2){// Создаем убывающую сеть (типа [9, 8, 7, 6, 5, 4]) с ограничениями по числу нейронов и слоев

N_current = N_max;
L_current = 0;

t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});" + "\n"// 2 inputs: x, y
for (x = 1; (x<=(L_max-L_min+1)); x++) {// Число слоев

	if (x==1){// Если слой первый, то число нейронов = макc
		N = N_current;
	}
	else{// Если не первый - вычисляем
		N_current = N_current - Math.ceil((N_max-N_min)/(L_max-L_min+1));
		if (N_current<1){
			N_current = 1;
		}
		else{
			N = N_current;
		}
	}

	total_neurons = total_neurons + N;
	
	if (total_neurons>neurons){//Отсечка по числу нейронов
		N = N - (total_neurons - neurons);
		total_neurons = neurons;	
		x = L_max;
	}

	if (N!=0){// Добавляем нейрон в сеть, только если он не нулевой
		net_param_temp = net_param_temp + N + ", ";
	}
	L_current = L_current + 1;// Увеличиваем текущее число слоев
  	t = t + "layer_defs.push({type:'fc', num_neurons:" + N + ", activation:'relu'});" + "\n";
}
t = t + "layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b \n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:step, momentum:mom, batch_size:batch, l2_decay:0.0});\n\
";
}



if (net_type == 3){// Создаем случайную сеть (типа [34, 2, 85, 11, 68]), с ограничениями по числу нейронов и слоев

L_current = 0;

t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});" + "\n"// 2 inputs: x, y
for (x = L_min; x < (L_min + L_max); x++) {// Тут создаем  случайное количество слоев со случайным количеством нейронов
	N = getRandomInt(N_min, N_max);
	if ((N + total_neurons) > neurons){// Перебор по числу нейронов
		N = neurons - total_neurons;
		total_neurons = neurons;
		x = L_max;
	}
	else{
		total_neurons = total_neurons + N;
	}
	if (N!=0){// Добавляем нейрон в сеть, только если он не нулевой
		net_param_temp = net_param_temp + N + ", ";
	}
	L_current = L_current + 1;// Увеличиваем текущее число слоев
	t = t + "layer_defs.push({type:'fc', num_neurons:" + N + ", activation:'relu'});" + "\n";
}
t = t + "layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b \n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:step, momentum:mom, batch_size:batch, l2_decay:0.0});\n\
";
}


}



/*
time_max_FPS_flag = 0;
net_type = 0;// Тип сети (фиксированнная)
create_neural_net(666, net_type);// Или сразу создаем сеть нужного типа и с нужным числом нейронов (тогда флаг обязательно в 0!)
*/



//////////////////////////////////PRESET FOR LONG QUALITY RENDER/////////////////////////////
flag_show_logo = 0;// Показываем или нет лого #neurodreams

L_min = 1;// Минимальное число слоев
L_current = L_min;// Текущее число слоев
L_max = 15;// Максимальное число слоев (15 - максимум?)

N_min = 80;// Минимальное число нейронов в слое
N_current = N_min;// Текущее число нейронов в слое
N_max = 80;// Максимальное число нейронов в слое

var sz_x = 500;// Выходное разрешение X
var sz_y = 500;// Выходное разрешение Y

var sz_g = Math.sqrt(sz_x*sz_x + sz_y*sz_y);// Размер гипотенузы (нужно для поворота картинки)
var sz_m = sz_g/sz_x;// Масштаб (нужно для поворота картинки)

time_max_FPS_flag = 0;
net_type = 1;// Тип сети (прямоугольная)

create_neural_net(1200, net_type);// Cоздаем сеть нужного типа и с нужным числом нейронов (тогда флаг обязательно в 0!)
//////////////////////////////////PRESET FOR LONG QUALITY RENDER/////////////////////////////


/*
///////////////////////////////////PRESET FOR SITE (adaptive)////////////////////////////////
flag_show_logo = 1;// Показываем или нет лого #neurodreams

L_min = 1;// Минимальное число слоев
L_current = L_min;// Текущее число слоев
L_max = 15;// Максимальное число слоев (15 - максимум?)

N_min = 10;// Минимальное число нейронов в слое
N_current = N_min;// Текущее число нейронов в слое
N_max = 30;// Максимальное число нейронов в слое

var sz_x = 500;// Выходное разрешение X
var sz_y = 500;// Выходное разрешение Y

var sz_g = Math.sqrt(sz_x*sz_x + sz_y*sz_y);// Размер гипотенузы (нужно для поворота картинки)
var sz_m = sz_g/sz_x;// Масштаб (нужно для поворота картинки)

time_max_FPS_flag = 1;
net_type = 1;// Тип сети (прямоугольная)

create_neural_net(1, net_type);// Cоздаем сеть нужного типа и с нужным числом нейронов (тогда флаг обязательно в 0!)
///////////////////////////////////PRESET FOR SITE (adaptive)////////////////////////////////
*/




/*
time_max_FPS_flag = 0;
net_type = 2;// Тип сети (треугольная, уменьшающаяся)
create_neural_net(500, net_type);// Создаем начальную сеть с одним слоем и одним нейроном и затем автоматически усложняем ее с учетом нужного FPS (тогда флаг обязательно в 1!)
*/

/*
time_max_FPS_flag = 0;
net_type = 3;// Тип сети (случайная)
create_neural_net(500, net_type);// Создаем начальную сеть с одним слоем и одним нейроном и затем автоматически усложняем ее с учетом нужного FPS (тогда флаг обязательно в 1!)
*/




function update() {//Собственно сама функция обработки

    // forward prop the data
    var W = nn_canvas.width;
    var H = nn_canvas.height;
    var p = oridata.data;
    var v = new convnetjs.Vol(1, 1, 2);
    var loss = 0;
    var lossi = 0;
    var N = batches_per_iteration;

    for (var iters = 0; iters < trainer.batch_size; iters++) {
        for (var i = 0; i < N; i++) {
            // sample a coordinate
            var x = convnetjs.randi(0, W);
            var y = convnetjs.randi(0, H);
            var ix = ((W * y) + x) * 4;
            var r = [p[ix] / 255.0, p[ix + 1] / 255.0, p[ix + 2] / 255.0]; // r g b
            v.w[0] = (x - W / 2) / W;
            v.w[1] = (y - H / 2) / H;
            var stats = trainer.train(v, r);
            loss += stats.loss;
            lossi += 1;
        }
    }
    loss /= lossi;

    if (counter === 0) smooth_loss = loss;
    else smooth_loss = 0.99 * smooth_loss + 0.01 * loss;

    var t = ''; 
    t += 'iteration: #' + counter + ' of ';
    $("#report").html(t);
   
    updateEveryIteration();
    //alert('iteration');

}



function draw() {// Отрисока картинки каждые mod_skip_draw итераций

    if (counter % mod_skip_draw !== 0) return;

    var W = nn_canvas.width;
    var H = nn_canvas.height;
    var g = nn_ctx.getImageData(0, 0, W, H);
    var v = new convnetjs.Vol(1, 1, 2);

    for (var x = 0; x < W; x++) {
        v.w[0] = (x - W / 2) / W;
        for (var y = 0; y < H; y++) {
            v.w[1] = (y - H / 2) / H;
            var ix = ((W * y) + x) * 4;
            var r = net.forward(v);
            g.data[ix + 0] = Math.floor(255 * r.w[0]);
            g.data[ix + 1] = Math.floor(255 * r.w[1]);
            g.data[ix + 2] = Math.floor(255 * r.w[2]);
            g.data[ix + 3] = 255; // alpha...
        }
    }
    nn_ctx.putImageData(g, 0, 0);  
	
    //Добавляем лого #neurodreams (если флаг установлен)
    if (flag_show_logo == 1){
    	font_shift = sz_x/40;
    	nn_ctx.fillStyle = "#787878";
    	//nn_ctx.font = "10pt Arial";
    	nn_ctx.textBaseline = "bottom";
    	nn_ctx.fillText("#neurodreams", font_shift + 5, sz_y - font_shift);
    }
  
}




function tick() {// Функция выполняющеяся каждую милисекунду

    update();
    draw();
    
    //Каждые 100 итераций сохраняем картинку
    if (counter % mod_skip_draw == 0 && counter != 0){
        
	time_finish = new Date();// Время окончания просчета картинки
	time_per_pic = (time_finish - time_start)/1000;
	// Генерируем строку параметров для имени файла
	//net_param = 'image ' + (files_current-1) + ', iteration ' + counter  + ', learning_rate ' + step + ' [0.01], momentum ' + mom + ' [0.9], batch_size ' + batch + ' [5], ' + L_current + ' layers [' + net_param_temp.substring(0, net_param_temp.length - 2) + ']' + ', ' + total_neurons + ' neurons' + ', time_per_pic = ' + time_per_pic.toFixed(1) + " sec.";// Все параметры
        net_param = 'image ' + (files_current-1) + ', iteration ' + counter  + ', ' + L_current + ' layers [' + net_param_temp.substring(0, net_param_temp.length - 2) + ']' + ', ' + total_neurons + ' neurons' + ', time_per_pic = ' + time_per_pic.toFixed(1) + " sec.";// Только самые важные
	
	saveImg(arrf[arr_x],counter);
	
	//document.title = (time_per_pic*(files_total-(files_current-1))*(iteration-counter)/100);// Остаток просчета текущей картинки (сек.)
	//document.title = ((time_finish - time_start)/1000) + " " + (files_current-1) + " of " + files_total + " " + (files_total)*(iteration)/100;// Temp
	$("#buttontp").val("time remaining: " + secToTime(Math.round(((files_total-(files_current-1))*(iteration/100) + (iteration-counter)/100)*(time_finish - time_start)/1000)));// Пишем оставшееся время на кнопке старта рендеринга (сек.)	
	time_start = new Date();// Новое время старта просчета
	time_current_FPS = 1/time_per_pic;// Измеряем текущий FPS

	if (time_current_FPS > time_max_FPS & (time_max_FPS_flag > 0)){ // Слишком быстро, усложняем сеть!

		//document.title = "+" + add_neurons + " neurons";
		total_neurons = total_neurons + add_neurons;
		
		//if (total_neurons <= max_neurons) ){// Если не превышены ограничения по размеру сети, то создаем новую сеть
			
			create_neural_net(total_neurons, net_type);// Создаем сеть с новыми параметрами
			$("#layerdef").val(t);// Выводим на экран новую сеть
			document.getElementById("p4").innerHTML = "rate: " + step + ", mom: " + mom + ", batch: " + batch + ", " + L_current + " layers [" + net_param_temp.substring(0, net_param_temp.length - 2) + "]" + ", " + total_neurons + " neurons" + " [+" + add_neurons + "]";// Обновляем параметры сети в строке параметров
			eval(t);// Запускаем новую сеть
		//}
		//else{
		//	time_max_FPS_flag = 0;// Сеть стала сложной (по ограничениям), сбрасываем флаг
		//	//total_neurons = max_neurons;
		//}

	}
	else{
		
		time_max_FPS_flag = 0;// Сеть стала считаться с нужным FPS, сбрасываем флаг
	
	}
	

	updateEveryStep();// Обновляем все на странице, в том числе fps
	
    }

    
    if(counter == iteration != 0){// Если закончили считать картинку – сохраняем ее и переходим к следующей
        saveImg(arrf[arr_x],counter);
        clearInterval(intervalID);
        arr_x += 1;
        reload();
    }

    counter += 1;
}




function reload() {// Запуск	

    counter = 0;

    //if (flag_heritage == 1){
    eval($("#layerdef").val());////////////////////////////////
    //}

    //alert(t);
   
    //alert('time_start');
    time_start = new Date();// Время старта просчета картинки
 

    if(arrf != undefined){
        var len_arr = arrf.length;
        files_total = len_arr;
	//alert('files_total');	
	if (files_current<=files_total){
		updateEveryStep();// Обновляем все что требуется на страничке		
		files_current = files_current + 1;
	}
    }else{
        return;
    }
    if(arr_x < len_arr){
        foreach_image(arr_x);
        intervalID = setInterval(tick, 1);
    }else{
        imgEnd();
    }
}





$(function () {

    var image = new Image();
       //alert('MAIN');

    image.onload = function () {
        ori_canvas = document.getElementById('canv_original');
        nn_canvas = document.getElementById('canv_net');

        drImgSize(sz_x, sz_y, image);

        //Старт обработки картинки
        //setInterval(tick, 1);

	eval(t);

    };

    //Картинка по умолчанию
    image.src = "imgs/logo.jpg";

    // init put text into textarea
    $("#layerdef").val(t);

    updateEveryStep();// Обновляем все что требуется на страничке

//Автозапуск при открытии страницы
    
    // Запуск с картинки
    $("#f").on('change', function (ev) {
        arrf = ev.target.files;
        var fr = new FileReader();
        var f = arrf[0];

        //alert('Запуск с картинки');
		
        fr.onload = function (ev2) {
            var image = new Image();
            image.onload = function () {

                //drImgSize(this.width, this.height, image);
                drImgSize(sz_x, sz_y, image);
                //sz_x = this.width;///
                //sz_y = this.height;///

                ori_ctx.drawImage(image, 0, 0, sz_x, sz_y);
                oridata = ori_ctx.getImageData(0, 0, sz_x, sz_y);
            };
		//alert('Начало замены image_src');
                image.src = ev2.target.result;
	        updateEveryStep();
		//alert('Конец замены image_src');

        };
        fr.readAsDataURL(f);
    });

    //Выбор картинки из галереи внизу
    /*
    $('.ci').click(function () {
        var src = $(this).attr('src');
        ori_ctx.drawImage(this, 0, 0, sz, sz);
        oridata = ori_ctx.getImageData(0, 0, sz, sz);
        reload();
    });
    */

    $('#buttontp_stop').click(function () {
        clearInterval(intervalID);
    });
});





function foreach_image(x) {

    var fr = new FileReader();
    var f = arrf[x];
    var grad;// На какой угол поворачиваем картинку

    //alert('Every image');

    fr.onload = function (ev2) {
        var image = new Image();
        image.onload = function () {

 	    
	    //alert('Катет = ' + sz_x + ' Гипотенуза = ' + sz_g + ' Масштаб = ' + sz_m);
            //sz_m = 1;
            ori_ctx.setTransform(sz_m, 0, 0, sz_m, 0, 0);// Изменяем масштаб картинки
            ori_ctx.translate((sz_x/2)-((sz_x/2)-(sz_x/2)/sz_m), (sz_y/2)-(sz_y/2-(sz_y/2)/sz_m));// Сдвигаем систему координат           
                	
	    if (flag_rotate == 1){// Поворачиваем или нет исходник
	    	grad = getRandomArbitary(-360,360);
	    	//document.title = grad.toFixed(0) + "°";// Пишем угол поворота в заголовке окна
	    	ori_ctx.rotate(inRad(grad));// Поворачиваем Картинку
	    }    

            ori_ctx.drawImage(image, -(sz_x)/2, -(sz_y)/2, sz_x, sz_y);// Отрисовываем картинку             
	    ori_ctx.translate(-(sz_x/2)-((sz_x/2)-(sz_x/2)/sz_m), -(sz_y/2)-(sz_y/2-(sz_y/2)/sz_m));// Сдвигаем систему координат обратно
            oridata = ori_ctx.getImageData(0, 0, sz_x, sz_y);// Текущая картинка - основа для следующей
            
        };
       
        //alert('Начало замены исходника image_src');
	image.src = ev2.target.result;//
	//alert('Конец замены исходника image_src');

    };
    fr.readAsDataURL(f);
}




function saveImg(arrf,iteration){
    var tmp = nn_canvas.toDataURL();
       //alert('saveImg');

    $.ajax({
        type: "POST",
        url: "lib.php",
        //data: "img="+tmp+"&name="+arrf.name+net_param+"&iteration="+iteration,
	data: "img="+tmp+"&name="+net_param,
        success: function(msg) {

        }
    });
}




function imgEnd(){//Закончили обрабатывать все фото, сохраняем в архив

   	window.location= "lib.php?end=1";

   	$("#f").removeAttr("disabled");// Делаем кнопку выбора файлов снова доступной
	$("#buttontp").val("choose some files");// Пишем новый текст на кнопке старта рендеринга
   	//$("#buttontp").removeAttr("disabled");// Включение кнопки
    	//$("#buttontp").attr(“disabled”,”disabled”);// Отключение кнопки
}




function drImgSize(szx, szy, image) {

    //alert('drImgSize');

    ori_canvas.width = szx;
    ori_canvas.height = szy;
    nn_canvas.width = szx;
    nn_canvas.height = szy;

    ori_ctx = ori_canvas.getContext("2d");
    nn_ctx = nn_canvas.getContext("2d");

    ori_ctx.drawImage(image, 0, 0, szx, szy);
    oridata = ori_ctx.getImageData(0, 0, szx, szy); // grab the data pointer. Our dataset.
    //alert('Теперь новая?');
  
}




function updateEveryIteration() {// Обновление параметров на странице каждую итерацию

	document.getElementById("p3").innerHTML="iteration: " + counter + "/" + iteration;// Пишем текущий номер итерации (1/500, например)
	//document.getElementById("p4").innerHTML = time_per_pic;
}



function updateEveryStep() {// Обновление параметров на странице каждый шаг (500 итераций по умолчанию)
	
	document.getElementById("p1").innerHTML="source image: " + (files_current) + "/" + files_total;// Пишем текущий номер обрабатываемой картинки (1/1, например)
	
	if (flag_debug == 1){// Отладка
		if (time_current_FPS > time_max_FPS){
			document.getElementById("p2").innerHTML = "fps:" + " " + time_current_FPS.toFixed(3) + " > " + time_max_FPS.toFixed(3) + ", time_max_FPS_flag" + " = " + time_max_FPS_flag;// Пишем текущий FPS и флаг в заголовке окна
		}
		else{
			document.getElementById("p2").innerHTML = "fps:" + " " + time_current_FPS.toFixed(3) + " < " + time_max_FPS.toFixed(3) + ", time_max_FPS_flag" + " = " + time_max_FPS_flag;// Пишем текущий FPS и флаг в заголовке окна	
		}
		document.getElementById("p4").innerHTML = "rate: " + step + ", mom: " + mom + ", batch: " + batch + ", " + L_current + " layers [" + net_param_temp.substring(0, net_param_temp.length - 2) + "]" + ", " + total_neurons + " neurons";
	}
	else{// Не отладка
		document.getElementById("p2").innerHTML = "fps:" + " " + time_current_FPS.toFixed(3);
		document.getElementById("p4").innerHTML = L_current + " layers [" + net_param_temp.substring(0, net_param_temp.length - 2) + "]" + ", " + total_neurons + " neurons";
	}

	document.getElementById("p3").innerHTML="iteration: " + counter + "/" + iteration;// Пишем текущий номер итерации (1/500, например)	
}




function PrefInt(number, len) {// Добавление нулей перед числом 00:00:32 например
	if ((String(number)).length<2){
   		return (Array(len).join('0') + number).slice(-length);
		}
	else{
		return (number);
	}
}



function secToTime(sec){// Перевод секунд в строку времени
    dt = new Date(); 
    dt.setTime(sec*1000); 
    return PrefInt(dt.getUTCHours(),2)+":"+PrefInt(dt.getUTCMinutes(),2)+":"+PrefInt(dt.getUTCSeconds(),2); 
} 



function inRad(num) {// Переводим градусы в радианы
	return num * Math.PI / 180;
}



function getRandomArbitary(min, max){// Случайное число между min и max
  return Math.random() * (max - min) + min;
}



function getRandomInt(min, max){// Случайное целое между min и max
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
