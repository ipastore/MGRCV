#ifndef TreeNode_H
#define TreeNode_H

#include <string>
#include <vector>
#include <geometry_msgs/PoseStamped.h>

class TreeNode
{
    private:
        std::vector <int> point;
        TreeNode *parent;
        std::vector<TreeNode *> children;

        int countNodesRec(TreeNode *root, int& count);

    public:
        TreeNode();
		TreeNode(std::vector <int> point_);
		~TreeNode();
		bool hasChildren();
        void appendChild(TreeNode *child);
        void setParent(TreeNode *parent);

        bool hasChildren() const { return children.size() > 0; }
        bool hasParent();

        TreeNode* getParent();
        TreeNode* getChild(int pos);

        int childrenNumber();


		std::vector <int> getNode();
		void printNode();
		void printTree();

		std::vector <std::vector <int>> returnSolution();
	
		TreeNode* nearNode(TreeNode* node1, TreeNode* node2);
		TreeNode* neast(TreeNode *root);
};

#endif
