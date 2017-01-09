#include "Base.h"

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
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
           w(i,j)+= alpha*(xi(i,j)-w(i,j));
		}
	  }        
	}


// Calcul de la distance entre une échantillon et un prototype
double distanceProto(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
	  double distance = 0.0;
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
          distance += (xi(i,j)-w(i,j))*(xi(i,j)-w(i,j));
				}
		}


	  return distance;
	}


// Calcul de la distance entre une échantillon et un prototype
// La distance est calculee entre le pixel du proto et une region de 3*3 autour du pixel de l'echantillon
double distanceProtoPonderee(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
	  double distance = 0.0;
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
          distance += 4*(xi(i,j)-w(i,j))*(xi(i,j)-w(i,j));
          // La distance n'est pas normé ici; on considere une perte L2
					if( i>0 && i<uci::Database::imagette::height && j>0 && j<uci::Database::imagette::width){
						for(int k = -1 ; k<2; ++k)
						    for(int l = -1; l<2; ++l)
								    if(!(k==0 && l==0))
										    distance += (xi(i+k,j+l)-w(i,j))*(xi(i+k,j+l)-w(i,j));
					}


				}
		}
	  return distance;
	}
// Calcul de la distance entre une échantillon et un prototype
// La distance est calculee entre le pixel du proto et une region de 3*3 moyenne
double distanceProtoMoyenne(const Prototypes::imagette& w, const uci::Database::imagette& xi)        {
	  double distance = 0.0;
		double pond = 0.0;
	  for(int i = 0 ; i < uci::Database::imagette::height ; ++i){
        for(int j = 0 ; j < uci::Database::imagette::width ; ++j){
          
          // La distance n'est pas normé ici; on considere une perte L2
					if( i>0 && i<uci::Database::imagette::height && j>0 && j<uci::Database::imagette::width){
						pond = 0.0;
						for(int k = -1 ; k<2; ++k)
						    for(int l = -1; l<2; ++l)
										    pond += xi(i+k,j+l);

						distance += (pond-w(i,j))*(pond-w(i,j));
					}else{
						distance += (xi(i,j)-w(i,j))*(xi(i,j)-w(i,j));
					}


				}
		}
	  return distance;
	}

// Trouve le prototype le plus proche d'un échantillon donné.	
void winnerProto(const Prototypes& protos, const uci::Database::imagette& xi, int& i, int& j) {
	for(int k =0; k < HEIGHT; ++k){
	   for(int l=0; l < WIDTH; ++l)
	      if (distanceProtoMoyenne(protos(i,j), xi) < distanceProtoMoyenne(protos(k,l), xi)){
	          i = k;
	          j = l;
		  }
	  }
	}


int main(int argc, char* argv[]) {
	
	Prototypes prototypes;
	
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
    
    double alpha = 0.35;
    int i = 0;
    int j = 0;
    
    int my_range = 100;
	prototypes = initProtos(database, HEIGHT, WIDTH);
	// K-means
	for(int p=0; p < my_range; ++p){
		//if(alpha > 0.15)
	    //	alpha = alpha - p*(0.003/my_range);
		//cout << alpha << endl;
	for(int m = 0; m < 100; ++m){
		  database.Next();
		  uci::Database::imagette& xi = database.input;
		  winnerProto(prototypes, xi, i, j);
		  learnProto(alpha, prototypes(i,j), xi);	
		}
		prototypes.PPM("proto",p);
	}
	
	
	return 0;
	
     
	}
