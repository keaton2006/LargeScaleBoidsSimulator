# LargeScaleBoidsSimulator  
本プログラムは，Reynoldsのボイドモデルの処理速度を,GPUによる並列処理で高速化したものです.  
具体的な仕様は以下のようになります.  
  1.２つのGPUの使用が前提で,片方を描画処理に,もう一方を計算処理に特化させています.  
  2.計算処理の並列化は,それぞれの粒子に対してスレッドを割り当てることで実現しています.  

＜主なソースファイルの説明＞  
1.gpu_controller.cu  
  2つのGPUへのスレッドの割当処理と,データの受け渡し方法の指定を行っています.  
2.calc.cu  
  計算処理を記述します.  
3.draw.cu  
  描画処理を記述します.  
4.param.h  
  パラメータを指定します.  
