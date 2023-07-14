# README: Standard Operating Procedure (SOP) for Desoldering Surface Mounted Components on PCBs for the fix in Opal Kelly Voltag Regulator

## Overview

This document provides the standard operating procedure for desoldering a surface-mounted component from the Opal Kelly FPGA Board (PCB). First the damaged component is identified. Then procedure utilizes flux and an air blower set to 300°C with a small nozzle to remove it. Afterwards, a new and compliant voltage regulator is soldered on to the open leads of the capacitors that corresponded to the desoldered component. Lastly, a hot glue gun is used to fix the component in place. Please ensure you read through and fully understand this SOP before beginning the process.

## Equipment

1. Soldering Station
2. Hot Air Blower with temperature control
3. Small Nozzle compatible with the Air Blower
4. Flux
5. Tweezers
6. Safety glasses
7. Heat-resistant mat or work surface
8. Soldering Iron
9. Solder
10. Replacement IC component
11. Multimeter

## Safety Precautions

- Always use safety glasses to protect your eyes from accidental flux or solder splashes.
- Use the heat-resistant mat or work surface to avoid heat damage.
- Handle the hot air blower and soldering iron with care to prevent burns.
- Make sure your working area is well-ventilated to avoid inhalation of flux or soldering fumes.
- Be careful to not touch and short any componetns while investigating which component needs replacing

## Procedure

1. **Preparation**: Remove the Dynap-SE2 from the other Opal Kelly board, it is not nessecary. Place the Opal Kelly board on a heat-resistant mat. Ensure your work area is clean and well-ventilated. 

2. **Identify the Component**: To identify the surface-mounted component that needs to be desoldered, you need to use the multimeter to measure the voltage potential across multiple capacitors when the board is ON. Make sure to be careful to not short any connection with the multimeter leads while underogoing this proces. Connect the Opal Kelly board to it's corresponding power supply now. 

    2. 1. **Opal Kelly Schematic** Utilize the following figure to identify all the capacitors to measure and the voltage potential that should be across each

    Insert Fig.1 here

    2. 2. **Multimeter measueremnr** Utilize the multimeter to measure the voltage across every capacitor. Be very cearful to only touch the metal parts across the capacitors. Identify which capacitor does not have the correct amount of voltage across it. 

    2. 3. **Component Identification** After you have ideintified the capacitor that does not have the correct amount of voltage dropping across it, you can then identify which IC needs to be replaced. Place the multimeter back in it's right place, we will now proceed to desolder this component.  

**UNPLUG THE OPAL KELLY BOARD BEFORE PROCEDING**

3. **Apply Flux**: Apply a small amount of flux to the solder joints of the damaged IC component. This will facilitate the melting of the solder and its removal from the joint.

4. **Hot Air Blower Setup**: Attach the small nozzle to the hot air blower. Set the temperature to 300°C. Note that many hot air blowers turn on and heat up automatically when removed from their base. Please be careful. 

5. **Desoldering**: Aim the hot air at the fluxed solder joint from a short distance (about 1-2 cm) away. Keep the blower moving in small circles around the component to evenly distribute the heat. Avoid focusing on one area for too long to prevent PCB or component damage.

6. **Component Removal**: Once the solder melts, carefully lift the component using a pair of tweezers. Apply gentle force - if the component does not lift easily, reapply heat until it does.

7. **Clean-Up**: After removal, clean the area with a solder wick to remove remaining solder, and then with isopropyl alcohol to remove flux residues. Inspect the PCB for any potential damage.

8. **Review and Verification**: Inspect the component and PCB under a microscope if possible, to ensure complete removal of the component and no damage to the PCB.

9. **Switch tools** Turn off the hot air and remove it, prepare and heat up a soldering iron and as well. Remove the new component from its packaging and have it ready.

10. **Identify placement of new component** Refer to figure 2 to acertain where this new component needs to be soldered too to properly power up the system. Depending on the component, differnt leads need to be soldered into differnt cap leads.

11. **Apply extra solder to the caps** to properly solder the components to the pcb, we need to first over-solder the exposed pads of the capacitors. Take your soldering iron and solder to add a small mountain of solder popping out of each component. Ths is done so in a later step you can solder the component and capacitor pads together by just applying heat to both with a solder.

11. **Bend the Leads** Most probably, the replacement component is a through hole component that needs to be soldered in an unconventional way. To do so take a moment to look a the legs and see if bending one of the legs could facilitate having all the legs align with all the components. Do this carefully, and slowly. 

12. **Solder** Now, take the component and push it's legs agains the component pads with the extra solder. Create connections by applying heat with the solder to the oversoldered pads and push the legs of the component into the the now liquified solder. After the legs are inside the pusslde of solder,remove the soldering iron and let the legs of the component settle into the now solidified solder. Do this for each of the legs. If you find that one of the legs is too short to reach the pad of one of the capacitors, you can create a connection by applying additional solder to the pad of the capacitor and essentialy builda brige towards the component leg. Once this process is complete, please turn off the Soldering Iron. 

13. **Review and Verification**: Inspect the component and PCB, remove solder that has been placed in incorrect places. Make sure no other component has been moved, damaged or desoldered during this process. You can now plug in the power supply into the opal kelly board to check if replacing the IC component has indeed returned the board to a functioning state. Go verify that the DynapSE2 and OpallKelly board now work properly together. If not, report this to the designated people. 

15. **Hot glue gun**: When soldering a dangling though hole component to a PCB, its imperative to glue it into place to avoid unessecary strain on the soldered connections due to mechanical stress. Use a hot glue gun to do so. Sodler the component body into a part of the PCB as in Fig.4. If glue from the glue gun ends up in extraneous parts of the PCB, remove it with the tweezers. 

15. **You are done**: If the replacement of the component has returned th OpalKelly board to a functioning state, and the components of the PCB are properly secured you have finished the procedure delinated by this document. Bathe in the satisfaction that you can now train SNN's without silly voltage regulators hindering your progess. 

16. **Clean**: No wait, you are not done, please clean everything you used and return all tools to their corresponding place. 


## Troubleshooting

If the solder does not melt as expected, reapply some flux and repeat the heating process. If the component still does not come off, verify that you're using the correct temperature and that the heat is being applied correctly. If issues persist, seek assistance from a more experienced technician.

## Conclusion

Desoldering surface mounted components is a delicate task that requires precision, patience, and practice. Always prioritize safety and remember that the goal is to remove the component without causing damage to the PCB or the component itself. If you're new to desoldering, you may wish to practice on a non-essential board first. Happy desoldering!
