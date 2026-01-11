#!/usr/bin/env python3
"""
OSM Web Wizard Parameter Calculator
Helps determine appropriate parameters for SUMO OSM Web Wizard
"""

def calculate_traffic_parameters(target_total_veh_h, truck_percentage=0.10, 
                                 through_factor=0.3, include_buses=False,
                                 include_motorcycles=False, include_pedestrians=False):
    """
    Calculate OSM Web Wizard parameters to achieve target traffic volume
    
    Args:
        target_total_veh_h: Target total vehicles per hour (after through traffic)
        truck_percentage: Percentage of trucks (0.10 = 10%)
        through_factor: Through traffic factor (0.3 = 30% additional)
        include_buses: Whether to include buses
        include_motorcycles: Whether to include motorcycles
        include_pedestrians: Whether to include pedestrians
        
    Returns:
        Dictionary with recommended parameters
    """
    # Calculate base traffic (before through factor)
    # target = base × (1 + through_factor)
    base_total = target_total_veh_h / (1 + through_factor)
    
    # Allocate to vehicle types
    # Assume: cars + trucks + buses + motorcycles = base_total
    # trucks = base_total × truck_percentage
    # cars = base_total × (1 - truck_percentage) - other_vehicles
    
    # Typical allocations
    bus_percentage = 0.02 if include_buses else 0.0  # 2% buses
    motorcycle_percentage = 0.05 if include_motorcycles else 0.0  # 5% motorcycles
    
    # Calculate vehicle counts
    trucks_h = int(base_total * truck_percentage)
    buses_h = int(base_total * bus_percentage) if include_buses else 0
    motorcycles_h = int(base_total * motorcycle_percentage) if include_motorcycles else 0
    
    # Remaining goes to cars
    other_vehicles = trucks_h + buses_h + motorcycles_h
    cars_h = int(base_total - other_vehicles)
    
    # Pedestrians (separate, doesn't affect vehicle count)
    pedestrians_h = int(cars_h * 0.1) if include_pedestrians else 0  # 10% of car count
    
    # Calculate actual total (for verification)
    base_vehicles = cars_h + trucks_h + buses_h + motorcycles_h
    actual_total = base_vehicles * (1 + through_factor)
    
    return {
        'cars_per_hour': cars_h,
        'trucks_per_hour': trucks_h,
        'buses_per_hour': buses_h,
        'motorcycles_per_hour': motorcycles_h,
        'pedestrians_per_hour': pedestrians_h,
        'through_traffic_factor': through_factor,
        'base_total_veh_h': base_vehicles,
        'expected_total_veh_h': actual_total,
        'target_total_veh_h': target_total_veh_h,
        'difference': abs(actual_total - target_total_veh_h)
    }


def print_recommendations():
    """Print recommended parameter sets for common scenarios"""
    
    scenarios = [
        {
            'name': 'Low Traffic (Baseline)',
            'target': 400,
            'truck_pct': 0.10,
            'through': 0.2,
            'buses': True,
            'motorcycles': True,
            'pedestrians': True
        },
        {
            'name': 'Moderate Traffic (Your Current)',
            'target': 1150,
            'truck_pct': 0.10,
            'through': 0.3,
            'buses': False,
            'motorcycles': False,
            'pedestrians': False
        },
        {
            'name': 'High Traffic (Stress Test)',
            'target': 2000,
            'truck_pct': 0.10,
            'through': 0.4,
            'buses': True,
            'motorcycles': True,
            'pedestrians': True
        },
        {
            'name': 'Urban Grid (Moderate)',
            'target': 1500,
            'truck_pct': 0.08,
            'through': 0.4,
            'buses': True,
            'motorcycles': True,
            'pedestrians': True
        },
        {
            'name': 'Highway/Arterial',
            'target': 3000,
            'truck_pct': 0.15,
            'through': 0.8,
            'buses': False,
            'motorcycles': False,
            'pedestrians': False
        }
    ]
    
    print("=" * 80)
    print("OSM WEB WIZARD PARAMETER RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    for scenario in scenarios:
        params = calculate_traffic_parameters(
            target_total_veh_h=scenario['target'],
            truck_percentage=scenario['truck_pct'],
            through_factor=scenario['through'],
            include_buses=scenario['buses'],
            include_motorcycles=scenario['motorcycles'],
            include_pedestrians=scenario['pedestrians']
        )
        
        print(f"Scenario: {scenario['name']}")
        print("-" * 80)
        print(f"  Target Total: {scenario['target']} veh/h")
        print(f"  Cars:          {params['cars_per_hour']} veh/h")
        print(f"  Trucks:        {params['trucks_per_hour']} veh/h ({scenario['truck_pct']*100:.0f}%)")
        if params['buses_per_hour'] > 0:
            print(f"  Buses:         {params['buses_per_hour']} veh/h")
        if params['motorcycles_per_hour'] > 0:
            print(f"  Motorcycles:   {params['motorcycles_per_hour']} veh/h")
        if params['pedestrians_per_hour'] > 0:
            print(f"  Pedestrians:   {params['pedestrians_per_hour']} ped/h")
        print(f"  Through Factor: {params['through_traffic_factor']}")
        print(f"  Expected Total: {params['expected_total_veh_h']:.0f} veh/h")
        print()
    
    print("=" * 80)
    print("QUICK REFERENCE")
    print("=" * 80)
    print("""
Through Traffic Factor Guidelines:
  - Single Intersection:    0.2 - 0.4
  - Urban Grid:             0.3 - 0.5
  - Highway/Arterial:       0.7 - 1.0
  - Residential:            0.1 - 0.2

Truck Percentage Guidelines:
  - Urban:                  5% - 10%
  - Industrial:             10% - 15%
  - Highway:                15% - 25%

Typical Vehicle Ratios:
  - Cars:                   85% - 95%
  - Trucks:                  5% - 15%
  - Buses:                   1% -  3%
  - Motorcycles:             2% -  5%
    """)


if __name__ == "__main__":
    print_recommendations()
    
    print("\n" + "=" * 80)
    print("CUSTOM CALCULATION")
    print("=" * 80)
    print("\nEnter your target traffic volume to get recommendations:")
    
    try:
        target = float(input("Target total vehicles/hour (after through traffic): "))
        truck_pct = float(input("Truck percentage (0.10 for 10%, default 0.10): ") or "0.10")
        through = float(input("Through traffic factor (0.3 for 30%, default 0.3): ") or "0.3")
        
        params = calculate_traffic_parameters(
            target_total_veh_h=target,
            truck_percentage=truck_pct,
            through_factor=through
        )
        
        print("\nRecommended OSM Web Wizard Parameters:")
        print(f"  Cars:          {params['cars_per_hour']} veh/h")
        print(f"  Trucks:        {params['trucks_per_hour']} veh/h")
        print(f"  Through Factor: {params['through_traffic_factor']}")
        print(f"  Expected Total: {params['expected_total_veh_h']:.0f} veh/h")
        print(f"  (Difference from target: {params['difference']:.0f} veh/h)")
        
    except (ValueError, KeyboardInterrupt):
        print("\nCalculation cancelled or invalid input.")


