/******************************************************************
 * ArduinoANN - An artificial neural network for the Arduino
 * All basic settings can be controlled via the Network Configuration
 * section.
 * See robotics.hobbizine.com/arduinoann.html for details.
 ******************************************************************/

#include <math.h>

#define BOTAO1 2
#define BOTAO2 3

#define LED1 4
#define LED2 5
#define LED3 6
#define LED4 7

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

const int PatternCount = 4;
const int InputNodes = 2;
const int HiddenNodes = 8;
const int OutputNodes = 4;
const float LearningRate = 0.3;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.004;

const byte Input[PatternCount][InputNodes] = {
  //{ 1, 1, 1, 1, 1, 1, 0 },  // 0
  //{ 0, 1, 1, 0, 0, 0, 0 },  // 1
  //{ 1, 1, 0, 1, 1, 0, 1 },  // 2
  //{ 1, 1, 1, 1, 0, 0, 1 },  // 3
  //{ 0, 1, 1, 0, 0, 1, 1 },  // 4
  //{ 1, 0, 1, 1, 0, 1, 1 },  // 5
  //{ 0, 0, 1, 1, 1, 1, 1 },  // 6
  //{ 1, 1, 1, 0, 0, 0, 0 },  // 7 
  //{ 1, 1, 1, 1, 1, 1, 1 },  // 8
  //{ 1, 1, 1, 0, 0, 1, 1 }   // 9
  {0,0},{0,1},{1,0},{1,1}
}; 

const byte Target[PatternCount][OutputNodes] = {
  { 0, 0, 0, 1 },  
  { 0, 0, 1, 0 }, 
  { 0, 1, 0, 0 }, 
  { 1, 0, 0, 0 }
//  { 0, 0, 0, 0 }, 
//  { 0, 0, 0, 1 }, 
//  { 0, 1, 1, 0 }, 
//  { 0, 1, 1, 1 }, 
//  { 1, 0, 0, 0 }, 
//  { 1, 0, 0, 1 } 
};

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;


float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

void setup(){
  Serial.begin(115200);
  randomSeed(analogRead(3));
  ReportEvery1000 = 1;
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p ;
  }

  pinMode(BOTAO1,INPUT_PULLUP);
  pinMode(BOTAO2,INPUT_PULLUP);

  pinMode(LED1,OUTPUT);
  pinMode(LED2,OUTPUT);
  pinMode(LED3,OUTPUT);
  pinMode(LED4,OUTPUT);

  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
}  

void loop (){


/******************************************************************
* Initialize HiddenWeights and ChangeHiddenWeights 
******************************************************************/

  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
/******************************************************************
* Initialize OutputWeights and ChangeOutputWeights
******************************************************************/

  for( i = 0 ; i < OutputNodes ; i ++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  //Serial.println("Initial/Untrained Outputs: ");
  Serial.println("Inicio/Saidas nao treinadas: ");
  toTerminal();
/******************************************************************
* Begin training 
******************************************************************/

  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    

/******************************************************************
* Randomize order of training patterns
******************************************************************/

    for( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
/******************************************************************
* Cycle through each training pattern in the randomized order
******************************************************************/
    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

/******************************************************************
* Compute hidden layer activations
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }


/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/


      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

      for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

/******************************************************************
* Every 1000 cycles send data to terminal for display
******************************************************************/
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      Serial.println(); 
      Serial.println(); 
      Serial.print ("Ciclo de treinamento(iteracao): ");
      Serial.print (TrainingCycle);
      Serial.print ("  Erro = ");
      Serial.println (Error, 5);

      toTerminal();

      if (TrainingCycle==1)
      {
        ReportEvery1000 = 999;
      }
      else
      {
        ReportEvery1000 = 1000;
      }
    }    


/******************************************************************
* If error rate is less than pre-determined threshold then end
******************************************************************/

    if( Error < Success ) break ;  
  }
  Serial.println ();
  Serial.println(); 
  Serial.print ("Ciclo de treinamento(iteracao): ");
  Serial.print (TrainingCycle);
  Serial.print ("  Erro = ");
  Serial.println (Error, 5);

  toTerminal();

  Serial.println ();  
  Serial.println ();
  Serial.println ("Treinamento bem sucedido! ");
  Serial.println ("--------"); 
  Serial.println ();
  Serial.println ();  
  ReportEvery1000 = 1;

  
  
  while(1){
    testando();
    delay(1000);
  }
}

void toTerminal()
{
  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Padrao de treinado: ");
    Serial.println (p);      
    Serial.print ("  Entrada ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Desejado ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Saida ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }


}

char incomingByte = ""; // for incoming serial data
bool leitura1 = 0;
bool leitura2 = 0;
bool leu = 0;

void testando(){

  //bool leitura1 = !digitalRead(BOTAO1);
  //bool leitura2 = !digitalRead(BOTAO2);

  

  while(Serial.available() > 0) {
    // read the incoming byte:
    incomingByte = Serial.read();

    if(!leu){
      if(incomingByte == '0'){
        leitura1 = 0;
      }
      else{
        leitura1 = 1;
      }
      leu = 1;
    }else{
      if(incomingByte == '0'){
        leitura2 = 0;
      }
      else{
        leitura2 = 1;
      }
      leu = 0;
    }
    
    // say what you got:
    //Serial.print("I received: ");
    //Serial.println(incomingByte);
  }

  int desejado = leitura1 << 1;
  desejado = desejado | leitura2;
  
  //Serial.println(); 
  //Serial.println(desejado);

  char InputZ[1][InputNodes] = {leitura1,leitura2};
  //InputZ[0,1] = (byte)1;
  
  for( p = 0 ; p < 1 ; p++ ) { 
    //Serial.println(); 
    //Serial.println();
    
    Serial.print (" Entrada ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (InputZ[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Desejado ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[desejado][i], DEC);
      Serial.print (" ");
    }
/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += InputZ[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("  Saida ");
    for( i = 0 ; i < OutputNodes ; i++ ) {

      bool acende = 0;
      if(Output[i] >0.5) acende = 1; else acende = 0;
      //Serial.print (Output[i], 5);

      if(i == 0){
        //digitalWrite(LED1,acende);
        Serial.print("LED1 = ");
      } else if(i == 1){
        //digitalWrite(LED2,acende);
        Serial.print(" ** LED2 = ");
      } else if(i == 2){
        //digitalWrite(LED3,acende);
        Serial.print(" ** LED3 = ");
      } else if(i == 3){
        //digitalWrite(LED4,acende);
        Serial.print(" ** LED4 = ");
      }
      
      Serial.print (Output[i],0);
      Serial.print(" ");
    }
    Serial.println();
    delay(1000);
  }
}


