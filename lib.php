<?php
if(isset($_POST['img'])){
    if (!is_dir('new_imgs')) {
        mkdir('new_imgs');
    }
    $img = $_POST['img'];
    $name = $_POST['name'];
    $iteration = (int)$_POST['iteration'];
    $img = str_replace('data:image/png;base64,', '', $img);
    $img = str_replace(' ', '+', $img);
    $result = file_put_contents('new_imgs/'.$name.'_'.$iteration.'.png', base64_decode($img));
}

if(isset($_GET['end'])){
    $zip_name = 'archive.zip';
    $src = 'new_imgs';
    $zip = new ZipArchive();
    $zip->open($zip_name, ZIPARCHIVE::CREATE);

    $dir = opendir($src);
    while(false !== ( $file = readdir($dir)) ) {
        if (( $file != '.' ) && ( $file != '..' )) {
            $full = $src . '/' . $file;
            if(!is_dir($full) ) {
                $zip->addFile($full); //Добавляем в архив файл
            }
        }
    }

    $zip->close(); //Завершаем работу с архивом

    if (file_exists($zip_name)) {
        // отдаём файл на скачивание
        header('Content-type: application/zip');
        header('Content-Disposition: attachment; filename="' . $zip_name . '"');
        readfile($zip_name);
        // удаляем zip файл если он существует
        sleep(2);
        unlink($zip_name);
    }

    rrmdir($src);
}

function rrmdir($src) {
    $dir = opendir($src);
    while(false !== ( $file = readdir($dir)) ) {
        if (( $file != '.' ) && ( $file != '..' )) {
            $full = $src . '/' . $file;
            if ( is_dir($full) ) {
                rrmdir($full);
            }
            else {
                unlink($full);
            }
        }
    }
    closedir($dir);
    rmdir($src);
}