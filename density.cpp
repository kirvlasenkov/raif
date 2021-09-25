#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>

using namespace std;

int main(){
    int n;
    scanf("%d", &n);
    vector<complex<float>> points(n);
    for(int i=0; i < n; i++) {
        float x, y;
        scanf("%f, %f", &x, &y);
        points[i] = {x, y};
    }

    vector<float> density(n, 0);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(j == i) continue;
            float dist = abs(points[i] - points[j]) * 111 / 2;
            density[i] += exp(-dist);
        }
    }

    for(int i = 0; i < n; i++) {
        printf("%f\n", density[i]);
    }
}