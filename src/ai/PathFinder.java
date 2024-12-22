package ai;

import java.util.ArrayList;

import Main.GamePanel;

public class PathFinder {

    GamePanel gp;
    Node[][] node;
    ArrayList<Node> openList = new ArrayList<>();
    public ArrayList<Node> pathList = new ArrayList<>();
    Node startNode, goalNode, currentNode;
    boolean goalReached = false;
    public int step = 0;

    public PathFinder(GamePanel gp) {

        this.gp = gp;
        instantiateNodes();
    }

    public void instantiateNodes() {

        node = new Node[gp.maxWorldColumn][gp.maxWorldRow];

        int col = 0;
        int row = 0;

        while (col < gp.maxWorldColumn && row < gp.maxWorldRow) {

            node[col][row] = new Node(col, row);

            col++;
            if (col == gp.maxWorldColumn) {
                col = 0;
                row++;
            }
        }
    }

    public void resetNodes() {
        int col = 0;
        int row = 0;

        while (col < gp.maxWorldColumn && row < gp.maxWorldRow) {

            // RESET OPEN, CHECKED AND SOLID STATE
            node[col][row].open = false;
            node[col][row].checked = false;
            node[col][row].solid = false;

            col++;
            if (col == gp.maxWorldColumn) {
                col = 0;
                row++;
            }
        }

        // RESET OTHER THINGS
        openList.clear();
        pathList.clear();
        goalReached = false;
        step = 0;
    }

    public void setNodes(int startCol, int startRow, int goalCol, int goalRow) {

        resetNodes();

        // SET START AND GOAL NODES
        startNode = node[startCol][startRow];
        currentNode = startNode;
        goalNode = node[goalCol][goalRow];
        openList.add(currentNode);

        int col = 0;
        int row = 0;

        while (col < gp.maxWorldColumn && row < gp.maxWorldRow) {

            // SET SOLID NODE
            // CHECK TILES
            int tileNum = gp.tileM.mapTileNum[gp.currentMap][col][row];
            if (gp.tileM.tile[tileNum].collision == true) {
                node[col][row].solid = true;
            }

            // CHECK INTERACTIVE TILES
            for (int i = 0; i < gp.iTile[1].length; i++) {
                if (gp.iTile[gp.currentMap][i] != null && gp.iTile[gp.currentMap][i].destructible == true) {
                    int itCol = gp.iTile[gp.currentMap][i].worldX / gp.tileSize;
                    int itRow = gp.iTile[gp.currentMap][i].worldY / gp.tileSize;
                    node[itCol][itRow].solid = true;

                }
            }
            // SET COST
            getCost(node[col][row]);

            col++;
            if (col == gp.maxWorldColumn) {
                col = 0;
                row++;
            }
        }

    }

    public void getCost(Node node) {

        // GET G COST (DISTANCE FROM START NODE)
        int xDistance = Math.abs(node.col - startNode.col);
        int yDistance = Math.abs(node.row - startNode.row);
        node.gCost = xDistance + yDistance;

        // GET H COST (THE DISTANCE FROM THE GOAL NODE)
        xDistance = Math.abs(node.col - goalNode.col);
        yDistance = Math.abs(node.row - goalNode.row);
        node.hCost = xDistance + yDistance;

        // THE F COST (TOTAL COST)
        node.fCost = node.gCost + node.hCost;
    }

    public boolean search() {

        while (goalReached == false && step < 500) {

            int col = currentNode.col;
            int row = currentNode.row;

            // Check the current node
            currentNode.checked = true;
            openList.remove(currentNode);

            // OPEN THE UP NODE
            if (row - 1 >= 0) {
                openNode(node[col][row - 1]);
            }
            // OPEN THE LEFT NODE
            if (col - 1 >= 0) {
                openNode(node[col - 1][row]);
            }
            // OPEN THE RIGHT NODE
            if (col + 1 < gp.maxWorldColumn) {
                openNode(node[col + 1][row]);
            }
            // OPEN THE DOWN NODE
            if (row + 1 < gp.maxWorldRow) {
                openNode(node[col][row + 1]);
            }

            // FIND THE BEST NODE
            int bestNodeIndex = 0;
            int bestNodeFcost = 999;

            for (int i = 0; i < openList.size(); i++) {

                // Check if the node's fCost is better
                if (openList.get(i).fCost < bestNodeFcost) {
                    bestNodeIndex = i;
                    bestNodeFcost = openList.get(i).fCost;
                }
                // If F cost is equal, check the G cost
                else if (openList.get(i).fCost == bestNodeFcost) {
                    if (openList.get(i).gCost < openList.get(bestNodeIndex).gCost) {
                        bestNodeIndex = i;
                    }
                }
            }
            // If there is no node in the openList, end the loop
            if (openList.size() == 0) {
                break;
            }

            // SET THE NEW CURRENT NODE
            currentNode = openList.get(bestNodeIndex);

            if (currentNode == goalNode) {
                goalReached = true;
                step = 0;
                trackThePath();
            }
            step++;
        }
        return goalReached;
    }

    public void openNode(Node node) {

        if (node.open == false && node.checked == false && node.solid == false) {

            // IF NODE NOT OPENED YET, ADD IT TO THE OPENED LIST
            node.open = true;
            node.parent = currentNode;
            openList.add(node);
        }
    }

    public void trackThePath() {
        // BACK TRACK AND DRAW THW BEST PATH
        Node current = goalNode;

        while (current != startNode) {

            pathList.add(0, current);
            current = current.parent;
        }
    }
}
