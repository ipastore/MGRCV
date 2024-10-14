#include <ros/ros.h>

int main(int argc, char** argv)
{
    // Inicializar el nodo ROS
    ros::init(argc, argv, "helloWorld");

    // Crear un NodeHandle, el punto de acceso para la comunicación con el sistema ROS
    ros::NodeHandle nh;

    // Definir la frecuencia de bucle (10 Hz)
    ros::Rate loop_rate(10);

    // Contador para el número de mensajes impresos
    unsigned int count = 0;

    // Bucle principal del nodo
    while (ros::ok())
    {
        // Imprimir "Hello World" y el número de mensaje
        ROS_INFO_STREAM("Hello World " << count);

        // Procesar mensajes entrantes (si los hay)
        ros::spinOnce();

        // Dormir para mantener la frecuencia de bucle a 10 Hz
        loop_rate.sleep();

        // Incrementar el contador
        count++;
    }

    return 0;
}
