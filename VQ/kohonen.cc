#include "Base.h"
#include <math.h>

// On va stocker les imagettes-prototypes au sein d'une grille.
#define WIDTH  10
#define HEIGHT 10
typedef uci::Map<WIDTH,HEIGHT,
		 uci::Database::imagette::width,
		 uci::Database::imagette::height> Prototypes;

// Cette fonction permet d'affecter un prototype (dont les pixels sont
// des double dans [0,255]) a une imagette tiree de la base (dont les
// pixels sont des unsigned char). Le & evite les copies inutiles.
void initProto(Prototypes::imagette& w,
               const uci::Database::imagette& xi) {
  //for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
    //for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
      //w(i,j) = (double)(xi(i,j));
}

// Initialisation d'une grille de prototype de taille height*width
Prototypes initProtos(uci::Database& database, int height, int width) {
	Prototypes prototypes;
	// Generation de height*width prototypes
	for(int k=0; k < height; ++k){
	  for(int l=0; l < width; ++l){
		  
	  	database.Next();

	    uci::Database::imagette& xi = database.input; // le & fait que xi est un pointeur, on evite une copie.
	    // Utilisons notre fonction de copie pour affecter un prototype à la valeur de l'imagette.
	    std::cout << "L'imagette tiree de la base est un " << database.what << std::endl;
	    initProto(prototypes(k,l),xi);
		  
	   }
    }
    return prototypes;
	}

// Mise à jour d'un prototype
void learnProto(double alpha, Prototypes::imagette& w, const uci::Database::imagette& xi){
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
           w(i,j)+= alpha*(xi(i,j)-w(i,j));
           
	}

// Utilise la distance dans le graphe
double moduleWinningRate(int& i_winner, int& j_winner, int& i, int& j, int& r){
	double graphDistance = sqrt((i_winner - i)*(i_winner - i) + (j_winner - j)*(j_winner - j));
	if(graphDistance < r){
		return (1-graphDistance/r);
	}else{
		return 0;
	}
        
}
// Mise à jour de plusieur prototypes
void learnProtos(double alpha, Prototypes& p, const uci::Database::imagette& xi, int& i_winner, int& j_winner, int& r){

	for(int k=0; k < HEIGHT; ++k){
	  for(int l=0; l < WIDTH; ++l){
		  double h = moduleWinningRate(i_winner, j_winner, k, l, r);
		  learnProto(h*alpha ,p(k,l), xi);
	  }
	}
           
	}

// Calcul de la distance entre une échantillon et un prototype
double distanceProto(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
	  double distance = 0.0;
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i)
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j)
		  distance += (xi(i,j)-w(i,j))*(xi(i,j)-w(i,j));
          // La distance n'est pas normé ici; on considere une perte L2
	  
	  return sqrt(distance);
	}

// Calcul de la distance entre une échantillon et un prototype
// La distance est calculee entre le pixel du proto et les pixels d'une region de 3*3 dont le pixel central a plus de poids
double distanceProtoPonderee(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
	  double distance = 0.0;
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
          distance += 4*(xi(i,j)-w(i,j))*(xi(i,j)-w(i,j));
          // La distance n'est pas normé ici; on considere une perte L2
					if( i>0 && i<uci::Database::imagette::height-1 && j>0 && j<uci::Database::imagette::width){
						for(int k = -1 ; k<2; ++k)
						    for(int l = -1; l<2; ++l)
								    if(!(k==0 && l ==0))
										    distance += (xi(i+k,j+l)-w(i,j))*(xi(i+k,j+l)-w(i,j));
					}
				}
		}
	  return distance;
	}

// Trouve le prototype le plus proche d'un échantillon donné.	
void winnerProto(const Prototypes& protos, const uci::Database::imagette& xi, int& i, int& j) {
	
	for(int k =0; k < HEIGHT; ++k){
	   for(int l=0; l < WIDTH; ++l)
	      if (distanceProtoPonderee(protos(i,j), xi) < distanceProtoPonderee(protos(k,l), xi)){
	          i = k;
	          j = l;
		  }
	  }
	}

int main(int argc, char* argv[]) {
	
	//Prototypes prototypes;
	
	uci::Database database;
	
	// Generation de 20 prototypes
	/*for(int k=0; k < HEIGHT; ++k){
	  for(int l=0; l < WIDTH; ++l){
		  
	  	database.Next();

	    uci::Database::imagette& xi = database.input; // le & fait que xi est un pointeur, on evite une copie.
	    // Utilisons notre fonction de copie pour affecter un prototype à la valeur de l'imagette.
	    //std::cout << "L'imagette tiree de la base est un " << database.what << std::endl;
	    initProto(prototypes(k,l),xi);
		  
	   }
    }*/
    
    double alpha = 0.65;
    int r = 3;
    int i_winner = 0;
    int j_winner = 0;

    
    int my_range = 100;
	Prototypes prototypes = initProtos(database, HEIGHT, WIDTH);
	// K-means
	for(int p=0; p < my_range; ++p){

	    // alpha += p*(0.8)/my_range;
	for(int m = 0; m < 50; ++m){
		  database.Next();
		  uci::Database::imagette& xi = database.input;
		  winnerProto(prototypes, xi, i_winner, j_winner);
		  learnProtos(alpha, prototypes, xi, i_winner, j_winner, r);	
		}
		prototypes.PPM("proto", p);
	}
	
	
	return 0;
	
     
	}
