#! /usr/bin/env python3

import gmsh
import sys
import numpy as np


class PolygonHole(object):

    def __init__(self, outer_vertices, inner_vertices, h_outer, h_inner):
        """
        Builds a mesh inside a polygon, and between the polygon and an
        outer rectangle

        Parameters
        ----------
        outer_vertices: array
            vertices of outer polygon
        inner_vertices: array
            vertices of hole
        h_outer: float
            target element size near outer vertices
        h_inner: float
            target element size near inner vertices
        """
        self.outer_vertices  = outer_vertices
        self.inner_vertices = inner_vertices
        self.h_outer = h_outer
        self.h_inner = h_inner



    def set_geometry(self, show=True):
        outer_vertices  = self.outer_vertices
        inner_vertices = self.inner_vertices
        h_outer = self.h_outer
        h_inner = self.h_inner

        # Initialize the API
        gmsh.initialize(sys.argv)
        gmsh.option.setNumber("General.Verbosity", 0)

        # Outer perimeter
        outer_list = []
        for v in outer_vertices:
            outer_list.append(gmsh.model.geo.addPoint(
                    v[0], v[1], 0., h_outer))

        outer_edges = []
        for i in range(len(outer_vertices)):
            outer_edges.append(gmsh.model.geo.addLine(
                    outer_list[i-1], outer_list[i]))

        gmsh.model.geo.addCurveLoop(outer_edges, 1)

        # Inner perimeter
        inner_list = []
        for v in inner_vertices:
            inner_list.append(gmsh.model.geo.addPoint(
                    v[0], v[1], 0., h_inner))

        inner_edges = []
        for i in range(len(inner_vertices)):
            inner_edges.append(gmsh.model.geo.addLine(
                    inner_list[i-1], inner_list[i]))

        gmsh.model.geo.addCurveLoop(inner_edges, 2)

        gmsh.model.geo.addPlaneSurface([1, 2], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)

        # Define boundary tags
        gmsh.model.addPhysicalGroup(1, outer_edges, 1)
        gmsh.model.setPhysicalName(1, 1, "Farfield")
        gmsh.model.addPhysicalGroup(1, inner_edges, 2)
        gmsh.model.setPhysicalName(1, 2, "Polygon")


        # Define area
        gmsh.model.addPhysicalGroup(2, [1], 3)
        gmsh.model.setPhysicalName(2, 3, "Outer Domain")
        gmsh.model.addPhysicalGroup(2, [2], 4)
        gmsh.model.setPhysicalName(2, 4, "Inner domain")

        gmsh.model.geo.synchronize()

        # Show the geometry (optional)
        if (show):
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()



    def mesh(self, filename, show=False, vtk=False):
        """
        Parameters
        ----------
        filename: string
            Name of output file (without extension)
        show: boolean
            Whether to show or not the produced mesh
        vtk: boolean
            Whether to save the mesh in vtk format or not
        """

        self.set_geometry(show=show)

        # # Compatibility with Capytaine
        # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)


        # Generate the mesh
        gmsh.model.mesh.generate(2)

        gmsh.write(filename + '.msh')

        if (vtk):
            gmsh.write(filename + '.vtk')

        # Show the geometry (optional)
        if (show):
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()

        gmsh.finalize()
