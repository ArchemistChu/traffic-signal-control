#!/usr/bin/env python3
"""
SUMO network file generator
Manually create a network file for a standard four-approach intersection
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_network_xml():
    """Create a SUMO network XML element"""
    
    # Create root element
    net = ET.Element("net", version="1.16", 
                     xmlns="http://sumo.dlr.de/xsd/net_file.xsd")
    
    # Add network location information
    location = ET.SubElement(net, "location", 
                           netOffset="0.00,0.00", 
                           convBoundary="-200.00,-200.00,200.00,200.00",
                           origBoundary="-200.00,-200.00,200.00,200.00",
                           projParameter="!")
    
    # Define edge type
    type_elem = ET.SubElement(net, "type", id="default", priority="1", 
                             numLanes="1", speed="13.89")
    
    # Add edge definitions
    edges = [
        {"id": "east_in", "from": "east_end", "to": "intersection", 
         "priority": "1", "numLanes": "4", "speed": "13.89",
         "shape": "200.00,0.00 0.00,0.00"},
        {"id": "west_in", "from": "west_end", "to": "intersection", 
         "priority": "1", "numLanes": "4", "speed": "13.89",
         "shape": "-200.00,0.00 0.00,0.00"},
        {"id": "north_in", "from": "north_end", "to": "intersection", 
         "priority": "1", "numLanes": "4", "speed": "13.89",
         "shape": "0.00,200.00 0.00,0.00"},
        {"id": "south_in", "from": "south_end", "to": "intersection", 
         "priority": "1", "numLanes": "4", "speed": "13.89",
         "shape": "0.00,-200.00 0.00,0.00"},
        {"id": "east_out", "from": "intersection", "to": "east_end", 
         "priority": "1", "numLanes": "1", "speed": "13.89",
         "shape": "0.00,0.00 200.00,0.00"},
        {"id": "west_out", "from": "intersection", "to": "west_end", 
         "priority": "1", "numLanes": "1", "speed": "13.89",
         "shape": "0.00,0.00 -200.00,0.00"},
        {"id": "north_out", "from": "intersection", "to": "north_end", 
         "priority": "1", "numLanes": "1", "speed": "13.89",
         "shape": "0.00,0.00 0.00,200.00"},
        {"id": "south_out", "from": "intersection", "to": "south_end", 
         "priority": "1", "numLanes": "1", "speed": "13.89",
         "shape": "0.00,0.00 0.00,-200.00"}
    ]
    
    for edge_data in edges:
        edge = ET.SubElement(net, "edge", **edge_data)
        
        # Add multiple lanes for incoming edges
        if "in" in edge_data["id"]:
            for i in range(int(edge_data["numLanes"])):
                lane = ET.SubElement(edge, "lane", 
                                   id=f"{edge_data['id']}_{i}",
                                   index=str(i),
                                   speed=edge_data["speed"],
                                   length="200.00",
                                   shape=edge_data["shape"],
                                   allow="passenger truck bus")
        else:
            # Outgoing edges have a single lane
            lane = ET.SubElement(edge, "lane", 
                               id=f"{edge_data['id']}_0",
                               index="0",
                               speed=edge_data["speed"],
                               length="200.00", 
                               shape=edge_data["shape"],
                               allow="passenger truck bus")
    
    # Add intersection (junction) definition
    junction = ET.SubElement(net, "junction", 
                           id="intersection",
                           type="traffic_light",
                           x="0.00", y="0.00",
                           incLanes="east_in_0 east_in_1 east_in_2 east_in_3 west_in_0 west_in_1 west_in_2 west_in_3 north_in_0 north_in_1 north_in_2 north_in_3 south_in_0 south_in_1 south_in_2 south_in_3",
                           intLanes="",
                           shape="-5.00,-5.00 -5.00,5.00 5.00,5.00 5.00,-5.00")
    
    # Add terminal (end) junctions
    end_junctions = [
        {"id": "east_end", "type": "dead_end", "x": "200.00", "y": "0.00"},
        {"id": "west_end", "type": "dead_end", "x": "-200.00", "y": "0.00"},
        {"id": "north_end", "type": "dead_end", "x": "0.00", "y": "200.00"},
        {"id": "south_end", "type": "dead_end", "x": "0.00", "y": "-200.00"}
    ]
    
    for junction_data in end_junctions:
        ET.SubElement(net, "junction", **junction_data)
    
    # Add connections (4 lanes: 0=right, 1=straight, 2=straight, 3=left)
    connections = [
        # Eastbound connections
        {"from": "east_in", "to": "south_out", "fromLane": "0", "toLane": "0", "dir": "r"},  # right turn
        {"from": "east_in", "to": "west_out", "fromLane": "1", "toLane": "0", "dir": "s"},   # straight
        {"from": "east_in", "to": "west_out", "fromLane": "2", "toLane": "0", "dir": "s"},   # straight
        {"from": "east_in", "to": "north_out", "fromLane": "3", "toLane": "0", "dir": "l"},  # left turn
        # Westbound connections
        {"from": "west_in", "to": "north_out", "fromLane": "0", "toLane": "0", "dir": "r"},  # right turn
        {"from": "west_in", "to": "east_out", "fromLane": "1", "toLane": "0", "dir": "s"},   # straight
        {"from": "west_in", "to": "east_out", "fromLane": "2", "toLane": "0", "dir": "s"},   # straight
        {"from": "west_in", "to": "south_out", "fromLane": "3", "toLane": "0", "dir": "l"},  # left turn
        # Northbound connections
        {"from": "north_in", "to": "west_out", "fromLane": "0", "toLane": "0", "dir": "r"},  # right turn
        {"from": "north_in", "to": "south_out", "fromLane": "1", "toLane": "0", "dir": "s"}, # straight
        {"from": "north_in", "to": "south_out", "fromLane": "2", "toLane": "0", "dir": "s"}, # straight
        {"from": "north_in", "to": "east_out", "fromLane": "3", "toLane": "0", "dir": "l"},  # left turn
        # Southbound connections
        {"from": "south_in", "to": "east_out", "fromLane": "0", "toLane": "0", "dir": "r"},  # right turn
        {"from": "south_in", "to": "north_out", "fromLane": "1", "toLane": "0", "dir": "s"}, # straight
        {"from": "south_in", "to": "north_out", "fromLane": "2", "toLane": "0", "dir": "s"}, # straight
        {"from": "south_in", "to": "west_out", "fromLane": "3", "toLane": "0", "dir": "l"}   # left turn
    ]
    
    for conn_data in connections:
        ET.SubElement(net, "connection", **conn_data)
    
    return net

def pretty_print_xml(element):
    """Pretty-print XML output"""
    rough_string = ET.tostring(element, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")[23:]  # remove the first XML declaration line

def main():
    """Main function"""
    print("Generating SUMO network file...")
    
    # Create network XML
    network_xml = create_network_xml()
    
    # Save to file
    with open("intersection.net.xml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(pretty_print_xml(network_xml))
    
    print("Network file generated: intersection.net.xml")
    print("File size:", os.path.getsize("intersection.net.xml"), "bytes")

if __name__ == "__main__":
    main() 