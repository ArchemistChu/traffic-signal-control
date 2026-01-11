#!/usr/bin/env python3
"""
Generate SUMO network files for 4 intersections
Creates node, edge, connection, and route files
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import random

def create_nodes_file():
    """Create nodes (junctions) file"""
    root = ET.Element("nodes")
    
    # 4 intersections in a grid
    intersections = [
        {"id": "TL1", "x": "-200", "y": "-200", "type": "traffic_light"},
        {"id": "TL2", "x": "200", "y": "-200", "type": "traffic_light"},
        {"id": "TL3", "x": "-200", "y": "200", "type": "traffic_light"},
        {"id": "TL4", "x": "200", "y": "200", "type": "traffic_light"},
    ]
    
    # External nodes for each intersection (entry/exit points)
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        x_base = -200 if "1" in tl_id or "3" in tl_id else 200
        y_base = -200 if "1" in tl_id or "2" in tl_id else 200
        
        external_nodes = [
            {"id": f"{tl_id}_east_end", "x": str(x_base + 200), "y": str(y_base), "type": "dead_end"},
            {"id": f"{tl_id}_west_end", "x": str(x_base - 200), "y": str(y_base), "type": "dead_end"},
            {"id": f"{tl_id}_north_end", "x": str(x_base), "y": str(y_base - 200), "type": "dead_end"},
            {"id": f"{tl_id}_south_end", "x": str(x_base), "y": str(y_base + 200), "type": "dead_end"},
        ]
        intersections.extend(external_nodes)
    
    for node in intersections:
        ET.SubElement(root, "node", **node)
    
    return root

def create_edges_file():
    """Create edges file"""
    root = ET.Element("edges")
    
    # For each intersection, create incoming and outgoing edges
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        x_base = -200 if "1" in tl_id or "3" in tl_id else 200
        y_base = -200 if "1" in tl_id or "2" in tl_id else 200
        
        # Incoming edges (2 lanes each)
        incoming_edges = [
            {"id": f"{tl_id}_east_in", "from": f"{tl_id}_east_end", "to": tl_id, "priority": "1", "numLanes": "2"},
            {"id": f"{tl_id}_west_in", "from": f"{tl_id}_west_end", "to": tl_id, "priority": "1", "numLanes": "2"},
            {"id": f"{tl_id}_north_in", "from": f"{tl_id}_north_end", "to": tl_id, "priority": "1", "numLanes": "2"},
            {"id": f"{tl_id}_south_in", "from": f"{tl_id}_south_end", "to": tl_id, "priority": "1", "numLanes": "2"},
        ]
        
        # Outgoing edges (1 lane each)
        outgoing_edges = [
            {"id": f"{tl_id}_east_out", "from": tl_id, "to": f"{tl_id}_east_end", "priority": "1", "numLanes": "1"},
            {"id": f"{tl_id}_west_out", "from": tl_id, "to": f"{tl_id}_west_end", "priority": "1", "numLanes": "1"},
            {"id": f"{tl_id}_north_out", "from": tl_id, "to": f"{tl_id}_north_end", "priority": "1", "numLanes": "1"},
            {"id": f"{tl_id}_south_out", "from": tl_id, "to": f"{tl_id}_south_end", "priority": "1", "numLanes": "1"},
        ]
        
        for edge in incoming_edges + outgoing_edges:
            ET.SubElement(root, "edge", **edge)
    
    # Connecting edges between intersections
    connecting_edges = [
        {"id": "TL1_to_TL2", "from": "TL1", "to": "TL2", "priority": "1", "numLanes": "1"},
        {"id": "TL2_to_TL1", "from": "TL2", "to": "TL1", "priority": "1", "numLanes": "1"},
        {"id": "TL3_to_TL4", "from": "TL3", "to": "TL4", "priority": "1", "numLanes": "1"},
        {"id": "TL4_to_TL3", "from": "TL4", "to": "TL3", "priority": "1", "numLanes": "1"},
        {"id": "TL1_to_TL3", "from": "TL1", "to": "TL3", "priority": "1", "numLanes": "1"},
        {"id": "TL3_to_TL1", "from": "TL3", "to": "TL1", "priority": "1", "numLanes": "1"},
        {"id": "TL2_to_TL4", "from": "TL2", "to": "TL4", "priority": "1", "numLanes": "1"},
        {"id": "TL4_to_TL2", "from": "TL4", "to": "TL2", "priority": "1", "numLanes": "1"},
    ]
    
    for edge in connecting_edges:
        ET.SubElement(root, "edge", **edge)
    
    return root

def create_connections_file():
    """Create connections file"""
    root = ET.Element("connections")
    
    # For each intersection, create connections
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        # Straight connections
        connections = [
            {"from": f"{tl_id}_east_in", "to": f"{tl_id}_west_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            {"from": f"{tl_id}_west_in", "to": f"{tl_id}_east_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            {"from": f"{tl_id}_north_in", "to": f"{tl_id}_south_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            {"from": f"{tl_id}_south_in", "to": f"{tl_id}_north_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            # Left turns
            {"from": f"{tl_id}_east_in", "to": f"{tl_id}_north_out", "fromLane": "1", "toLane": "0", "dir": "l"},
            {"from": f"{tl_id}_west_in", "to": f"{tl_id}_south_out", "fromLane": "1", "toLane": "0", "dir": "l"},
            {"from": f"{tl_id}_north_in", "to": f"{tl_id}_east_out", "fromLane": "1", "toLane": "0", "dir": "l"},
            {"from": f"{tl_id}_south_in", "to": f"{tl_id}_west_out", "fromLane": "1", "toLane": "0", "dir": "l"},
            # Right turns
            {"from": f"{tl_id}_east_in", "to": f"{tl_id}_south_out", "fromLane": "0", "toLane": "0", "dir": "r"},
            {"from": f"{tl_id}_west_in", "to": f"{tl_id}_north_out", "fromLane": "0", "toLane": "0", "dir": "r"},
            {"from": f"{tl_id}_north_in", "to": f"{tl_id}_west_out", "fromLane": "0", "toLane": "0", "dir": "r"},
            {"from": f"{tl_id}_south_in", "to": f"{tl_id}_east_out", "fromLane": "0", "toLane": "0", "dir": "r"},
        ]
        
        # Connections to adjacent intersections
        if tl_id == 'TL1':
            connections.extend([
                {"from": f"{tl_id}_east_in", "to": "TL1_to_TL2", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL2_to_TL1", "to": f"{tl_id}_west_out", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": f"{tl_id}_south_in", "to": "TL1_to_TL3", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL3_to_TL1", "to": f"{tl_id}_north_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            ])
        elif tl_id == 'TL2':
            connections.extend([
                {"from": f"{tl_id}_west_in", "to": "TL2_to_TL1", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL1_to_TL2", "to": f"{tl_id}_east_out", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": f"{tl_id}_south_in", "to": "TL2_to_TL4", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL4_to_TL2", "to": f"{tl_id}_north_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            ])
        elif tl_id == 'TL3':
            connections.extend([
                {"from": f"{tl_id}_east_in", "to": "TL3_to_TL4", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL4_to_TL3", "to": f"{tl_id}_west_out", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": f"{tl_id}_north_in", "to": "TL3_to_TL1", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL1_to_TL3", "to": f"{tl_id}_south_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            ])
        elif tl_id == 'TL4':
            connections.extend([
                {"from": f"{tl_id}_west_in", "to": "TL4_to_TL3", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL3_to_TL4", "to": f"{tl_id}_east_out", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": f"{tl_id}_north_in", "to": "TL4_to_TL2", "fromLane": "0", "toLane": "0", "dir": "s"},
                {"from": "TL2_to_TL4", "to": f"{tl_id}_south_out", "fromLane": "0", "toLane": "0", "dir": "s"},
            ])
        
        for conn in connections:
            ET.SubElement(root, "connection", **conn)
    
    return root

def create_routes_file():
    """Create routes file with sufficient vehicles"""
    root = ET.Element("routes", xmlns="http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Vehicle types
    vtypes = [
        {"id": "passenger", "accel": "2.6", "decel": "4.5", "sigma": "0.5", 
         "length": "5.0", "minGap": "2.5", "maxSpeed": "50", "color": "yellow"},
        {"id": "truck", "accel": "1.3", "decel": "4.0", "sigma": "0.5", 
         "length": "12.0", "minGap": "3.0", "maxSpeed": "40", "color": "red"},
        {"id": "bus", "accel": "1.2", "decel": "4.0", "sigma": "0.5", 
         "length": "18.0", "minGap": "3.5", "maxSpeed": "35", "color": "blue"},
    ]
    
    for vtype in vtypes:
        ET.SubElement(root, "vType", **vtype)
    
    # Define routes between intersections
    routes = []
    
    # Routes within each intersection (local traffic)
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        routes.extend([
            {"id": f"{tl_id}_east_to_west", "edges": f"{tl_id}_east_in {tl_id}_west_out"},
            {"id": f"{tl_id}_east_to_north", "edges": f"{tl_id}_east_in {tl_id}_north_out"},
            {"id": f"{tl_id}_east_to_south", "edges": f"{tl_id}_east_in {tl_id}_south_out"},
            {"id": f"{tl_id}_west_to_east", "edges": f"{tl_id}_west_in {tl_id}_east_out"},
            {"id": f"{tl_id}_west_to_north", "edges": f"{tl_id}_west_in {tl_id}_north_out"},
            {"id": f"{tl_id}_west_to_south", "edges": f"{tl_id}_west_in {tl_id}_south_out"},
            {"id": f"{tl_id}_north_to_south", "edges": f"{tl_id}_north_in {tl_id}_south_out"},
            {"id": f"{tl_id}_north_to_east", "edges": f"{tl_id}_north_in {tl_id}_east_out"},
            {"id": f"{tl_id}_north_to_west", "edges": f"{tl_id}_north_in {tl_id}_west_out"},
            {"id": f"{tl_id}_south_to_north", "edges": f"{tl_id}_south_in {tl_id}_north_out"},
            {"id": f"{tl_id}_south_to_east", "edges": f"{tl_id}_south_in {tl_id}_east_out"},
            {"id": f"{tl_id}_south_to_west", "edges": f"{tl_id}_south_in {tl_id}_west_out"},
        ])
    
    # Routes between intersections
    inter_routes = [
        # Horizontal routes
        {"id": "TL1_to_TL2", "edges": "TL1_east_in TL1_to_TL2 TL2_west_out"},
        {"id": "TL2_to_TL1", "edges": "TL2_west_in TL2_to_TL1 TL1_east_out"},
        {"id": "TL3_to_TL4", "edges": "TL3_east_in TL3_to_TL4 TL4_west_out"},
        {"id": "TL4_to_TL3", "edges": "TL4_west_in TL4_to_TL3 TL3_east_out"},
        # Vertical routes
        {"id": "TL1_to_TL3", "edges": "TL1_south_in TL1_to_TL3 TL3_north_out"},
        {"id": "TL3_to_TL1", "edges": "TL3_north_in TL3_to_TL1 TL1_south_out"},
        {"id": "TL2_to_TL4", "edges": "TL2_south_in TL2_to_TL4 TL4_north_out"},
        {"id": "TL4_to_TL2", "edges": "TL4_north_in TL4_to_TL2 TL2_south_out"},
        # Diagonal routes
        {"id": "TL1_to_TL4", "edges": "TL1_east_in TL1_to_TL2 TL2_south_in TL2_to_TL4 TL4_west_out"},
        {"id": "TL4_to_TL1", "edges": "TL4_west_in TL4_to_TL3 TL3_north_in TL3_to_TL1 TL1_south_out"},
        {"id": "TL2_to_TL3", "edges": "TL2_west_in TL2_to_TL1 TL1_south_in TL1_to_TL3 TL3_north_out"},
        {"id": "TL3_to_TL2", "edges": "TL3_east_in TL3_to_TL4 TL4_north_in TL4_to_TL2 TL2_south_out"},
    ]
    routes.extend(inter_routes)
    
    # Add routes to XML
    for route in routes:
        ET.SubElement(root, "route", **route)
    
    # Generate vehicle flows (sufficient traffic)
    # Local traffic at each intersection
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        for route_id in [f"{tl_id}_east_to_west", f"{tl_id}_west_to_east", 
                        f"{tl_id}_north_to_south", f"{tl_id}_south_to_north"]:
            ET.SubElement(root, "flow", 
                         id=f"flow_{route_id}",
                         type="passenger",
                         route=route_id,
                         begin="0",
                         end="3600",
                         period=str(random.randint(8, 15)),  # Vehicle every 8-15 seconds
                         departLane="best",
                         departSpeed="max")
    
    # Inter-intersection traffic
    for route_id in ["TL1_to_TL2", "TL2_to_TL1", "TL3_to_TL4", "TL4_to_TL3",
                    "TL1_to_TL3", "TL3_to_TL1", "TL2_to_TL4", "TL4_to_TL2",
                    "TL1_to_TL4", "TL4_to_TL1", "TL2_to_TL3", "TL3_to_TL2"]:
        ET.SubElement(root, "flow",
                     id=f"flow_{route_id}",
                     type="passenger",
                     route=route_id,
                     begin="0",
                     end="3600",
                     period=str(random.randint(12, 25)),  # Vehicle every 12-25 seconds
                     departLane="best",
                     departSpeed="max")
    
    return root

def create_traffic_lights_file():
    """Create traffic lights configuration file"""
    root = ET.Element("additional", xmlns="http://sumo.dlr.de/xsd/additional_file.xsd")
    
    # Standard 4-phase traffic light program for each intersection
    phases = [
        {"state": "GGGgrrrrGGGgrrrr", "duration": "30"},  # Phase 0: East-West green
        {"state": "yyyyrrrryyyyrrrr", "duration": "5"},    # Phase 1: Yellow
        {"state": "rrrrGGGgrrrrGGGg", "duration": "30"},   # Phase 2: North-South green
        {"state": "rrrryyyyrrrryyyy", "duration": "5"},   # Phase 3: Yellow
    ]
    
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        tl_logic = ET.SubElement(root, "tlLogic",
                                id=tl_id,
                                type="static",
                                programID="0",
                                offset="0")
        
        for phase in phases:
            ET.SubElement(tl_logic, "phase", **phase)
    
    return root

def pretty_print_xml(element):
    """Pretty-print XML"""
    rough_string = ET.tostring(element, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")[23:]

def main():
    """Main function"""
    print("Generating 4-intersection SUMO network files...")
    
    # Create and save files
    files = {
        "4intersection.nod.xml": create_nodes_file(),
        "4intersection.edg.xml": create_edges_file(),
        "4intersection.con.xml": create_connections_file(),
        "4intersection_routes.rou.xml": create_routes_file(),
        "4intersection_traffic_lights.add.xml": create_traffic_lights_file(),
    }
    
    for filename, root in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(pretty_print_xml(root))
        print(f"Generated: {filename} ({os.path.getsize(filename)} bytes)")
    
    print("\nNext step: Run netconvert to generate network file:")
    print("  netconvert -n 4intersection.nod.xml -e 4intersection.edg.xml -x 4intersection.con.xml -o 4intersection.net.xml")

if __name__ == "__main__":
    main()

