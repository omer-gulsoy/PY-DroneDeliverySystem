# English
## School Automation Project
***This project aims to optimize the delivery routes of drones within a certain geographic area, considering multiple
constraints such as drone battery life, maximum load capacity, no-fly zones, and delivery time windows. The system 
uses an A pathfinding algorithm combined with a genetic algorithm to assign delivery points efficiently to multiple 
drones, minimizing energy consumption and travel distance while maximizing the number of completed deliveries.****

***The project handles different drone characteristics, delivery priorities, and dynamically adjusts routes to 
avoid restricted flight zones active during certain times.***

### General Features:

    

    - Each drone has properties like maximum weight capacity, battery level, speed, and starting location.

    - Delivery points include location, package weight, priority, and delivery time windows.

    - Flight no-fly zones are defined by polygons and active time intervals, which drones must avoid.

    - The A* algorithm calculates the shortest valid path between points avoiding no-fly zones.

    - A genetic algorithm optimizes delivery assignments and routes across all drones over multiple generations.

    - Performance analysis includes delivery completion rate, average energy consumption, total traveled distance, 
    and execution time.

    - Visualization of drone routes, delivery points, and no-fly zones on a 2D map.

### Programming Languages and Tools ​​I Used:

    Backend     - Python 3.x
    
    Libraries:  - matplotlib (for visualization)
                - heapq (priority queue for A*)
                - datetime
                - random
                - math

> This project serves as a simulation and optimization tool to improve autonomous drone deliveries, balancing operational constraints and maximizing efficiency. 

> The texts in the program are in Turkish.

# Türkçe
## Drone Teslimat Optimizasyonu Projesi
***Bu proje, belirli bir coğrafi alanda dronların teslimat rotalarını optimize etmeyi amaçlamaktadır. Drone batarya 
kapasitesi, maksimum yük sınırı, uçuş yasak bölgeleri ve teslimat zaman aralıkları gibi kısıtlamalar dikkate alınarak, 
A algoritması ile güvenli rotalar hesaplanmakta, genetik algoritma ile teslimat noktalarının dronlar arasında en uygun 
şekilde dağıtımı sağlanmaktadır.***

***Proje, farklı dron özellikleri, teslimat öncelikleri ve zaman tabanlı yasak bölgeler gibi dinamik unsurları 
da yöneterek, enerji tüketimini ve seyahat mesafesini minimize ederken, tamamlanan teslimat sayısını maksimize etmeyi 
hedefler.***

### Genel Özellikler:

    - Her drone için maksimum taşıma ağırlığı, batarya kapasitesi, hız ve başlangıç konumu tanımlıdır.

    - Teslimat noktaları konum, paket ağırlığı, öncelik ve teslimat için zaman aralıkları bilgilerini içerir.

    - Uçuş yasak bölgeleri poligonlar ve aktif zaman dilimleri olarak tanımlanır ve dronlar tarafından aşılmamalıdır.

    - A* algoritması, no-fly bölgeleri göz önünde bulundurarak iki nokta arasındaki en uygun yolu hesaplar.

    - Genetik algoritma, tüm dronlar için teslimat noktalarını optimize ederek en iyi rota dağılımını sağlar.

    - Performans analizinde teslimat yüzdesi, ortalama enerji tüketimi, toplam mesafe ve çalışma süresi raporlanır.

    - 2 boyutlu harita üzerinde dron rotaları, teslimat noktaları ve yasak bölgeler görselleştirilir.

### Kullanılan Programlama Dilleri ve Araçlar:

    Backend:      - Python 3.x

    Kütüphaneler:   - matplotlib (görselleştirme)
                    - heapq (öncelik kuyruğu)
                    - datetime
                    - random
                    - math

> Bu proje, otonom drone teslimat sistemlerinin etkinliğini artırmak için kısıtları dikkate alan simülasyon ve optimizasyon aracı olarak geliştirilmiştir.
